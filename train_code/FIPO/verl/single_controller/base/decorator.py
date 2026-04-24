# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from functools import partial, wraps
from types import FunctionType

from verl.protocol import DataProtoFuture, _padding_size_key
from verl.utils.py_functional import DynamicEnum

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(DynamicEnum):
    """Enum class defining different dispatch modes for distributed computation.

    Each mode represents a specific strategy for distributing data across
    different ranks in a distributed system. The modes are used to control
    how data is partitioned and processed across different worker groups.
    """

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


class Execute(DynamicEnum):
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
init_predefined_dispatch_mode()
init_predefined_execute_mode()


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    for arg in args:
        assert isinstance(arg, DataProto | DataProtoFuture)
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, DataProto | DataProtoFuture)
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    data_proto_len = None
    padding_size = None

    def _padding_and_split_data(obj, chunks):
        nonlocal data_proto_len, padding_size
        assert isinstance(obj, DataProto | DataProtoFuture)
        if isinstance(obj, DataProto) and obj.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(obj)
                padding_size = (
                    (chunks - (data_proto_len % chunks))
                    if (data_proto_len % chunks > 0)
                    else 0
                )
            else:
                assert data_proto_len == len(
                    obj
                ), f"expecting all arg share same length of {data_proto_len}, but got {len(obj)}"
            obj.padding(padding_size=padding_size)
        return obj.chunk(chunks=chunks)

    splitted_args = [_padding_and_split_data(arg, chunks) for arg in args]
    splitted_kwargs = {
        key: _padding_and_split_data(val, chunks) for key, val in kwargs.items()
    }
    if padding_size is not None:
        splitted_kwargs[_padding_size_key] = padding_size

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def _concat_data_proto_or_future(output: list):
    import ray

    from verl.protocol import DataProto, DataProtoFuture

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    o = output[0]

    if isinstance(o, DataProto):
        return DataProto.concat(output)
    elif isinstance(o, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        worker_group.world_size, *args[1:], **kwargs
    )
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    import ray

    from verl.protocol import DataProto

    for o in output:
        assert isinstance(
            o, DataProto | ray.ObjectRef
        ), f"expecting {o} to be DataProto, but got {type(o)}"

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_nd_compute(
    dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs
):
    import ray

    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    args = [[ray.put(dp_arg) for dp_arg in arg] for arg in args]
    kwargs = {k: [ray.put(dp_v) for dp_v in v] for k, v in kwargs.items()}

    all_args = []
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_nd_compute(collect_mask: list[bool], worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size

    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        collect_dp_rank = collect_mask[global_rank]
        if collect_dp_rank:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_nd_compute_dataproto(
    dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs
):
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        dp_size, *args, **kwargs
    )
    return dispatch_nd_compute(
        dp_rank_mapping, dp_size, worker_group, *splitted_args, **splitted_kwargs
    )


def collect_nd_compute_dataproto(collect_mask: list[bool], worker_group, output):
    output = collect_nd_compute(collect_mask, worker_group, output)
    import ray

    from verl.protocol import DataProto

    for o in output:
        assert isinstance(
            o, DataProto | ray.ObjectRef
        ), f"expecting {o} to be DataProto, but got {type(o)}"
    return _concat_data_proto_or_future(output)


def dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # query dispatch info of the worker group
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(
            mesh_name
        )
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    # perform dispatch
    dp_size = max(dp_rank_mapping) + 1
    return dispatch_nd_compute_dataproto(
        dp_rank_mapping, dp_size, worker_group, *args, **kwargs
    )


def collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # the dispatch info is stored in the worker group
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(
            mesh_name
        )
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    # a boolean of whether the dp_rank is used for collect
    collect_mask = worker_group._collect_info[mesh_name]
    # perform dispatch
    return collect_nd_compute_dataproto(collect_mask, worker_group, *args, **kwargs)


def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    return {
        "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
        "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
    }


# Global registry for dispatch mode.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE: {
        "dispatch_fn": dispatch_dp_compute,
        "collect_fn": collect_dp_compute,
    },
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute,
    },
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert (
        dispatch_mode not in DISPATCH_MODE_FN_REGISTRY
    ), f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {
        "dispatch_fn": dispatch_fn,
        "collect_fn": collect_fn,
    }


def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert (
        dispatch_mode in DISPATCH_MODE_FN_REGISTRY
    ), f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {
        "dispatch_fn": dispatch_fn,
        "collect_fn": collect_fn,
    }


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(
        dispatch_mode, Dispatch | dict
    ), f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert (
                key in dispatch_mode
            ), f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(
        execute_mode, Execute
    ), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


"""
它的核心哲学是：业务逻辑与分布式逻辑解耦。

开发者只需关注函数内部的计算逻辑（“做什么”），而通过装饰器参数告诉框架如何处理数据的分发和同步（“怎么做”）。

这是装饰器的工厂函数，它允许你为被装饰的方法配置“元数据”：
- dispatch_mode: 决定数据怎么分。例如是广播给所有人（ONE_TO_ALL），还是切分给不同人（ALL_TO_ALL）。
- execute_mode: 决定谁去执行。是所有 Worker 都跑一遍（ALL），还是只有主节点跑（RANK_ZERO）。
- blocking: 决定调用后是否等待结果。
- materialize_futures: 这是一个高级优化选项，用于处理异步对象（Ray Future）。


这个 register 函数通过闭包和元编程，实现了以下流程：
接收配置：用户指定怎么分发数据。
预处理数据：在执行前自动把异步 Future 转为实体数据。
打标签：把配置信息绑定在函数属性上，供框架后续读取和调度。
这使得 VeRL 的代码看起来非常像单机代码，但背后却拥有强大的分布式调度能力。
"""


def register(
    dispatch_mode=Dispatch.ALL_TO_ALL,
    execute_mode=Execute.ALL,
    blocking=True,
    materialize_futures=True,
):
    """Register a function with distributed execution configuration. 注册一个函数，并配置分布式执行。

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.

    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    """
    包装函数：处理异步与数据“变现”

    这里定义了真正的包装逻辑，它做了两件事：
    1. 数据“变现” (_materialize_futures)：
        - 在分布式系统中，参数可能是一个“承诺”（Future/Promise），代表一个还没计算完的结果。
        - 如果 materialize_futures=True，装饰器会在执行函数前，强制等待这些 Future 完成，把数据真正取出来（即“变现”），确保函数内部拿到的是实实在在的数据，而不是句柄。
    2. 兼容异步：
        - 代码检测了原函数是否是 async 的（inspect.iscoroutinefunction），并分别生成了同步版 inner 和异步版 async_inner 包装器，保证框架能同时支持同步和异步代码。


    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        """
        注入“魔法”：元数据标记

        这是整个装饰器最精髓的部分：
        - MAGIC_ATTR：这是一个特殊的字符串常量（如 "attrs_3141562937"）。
        - setattr(...)：装饰器并没有直接执行分布式逻辑，而是把你配置的参数（分发模式、执行模式等）打包成一个字典，像贴标签一样贴在了函数对象上。
        
        为什么要这么做？
        - 这是一种元编程技巧。VeRL 的框架层（如 RayWorkerGroup）在初始化时，会扫描所有 Worker 类的方法。
          一旦发现某个方法身上有这个 MAGIC_ATTR 标签，框架就知道：“哦，这个方法需要分布式处理”，
          然后框架会根据标签里的配置，动态生成相应的分发和收集代码。
        """
        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {
            "dispatch_mode": dispatch_mode,
            "execute_mode": execute_mode,
            "blocking": blocking,
        }
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
