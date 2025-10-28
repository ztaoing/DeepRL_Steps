import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="这是一个示例程序")

# 添加参数
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='一个整数列表')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='将整数相加 (默认: 找出最大值)')

# 解析参数
args = parser.parse_args()

# 使用解析的参数
if args.accumulate == sum:
    result = sum(args.integers)
else:
    result = max(args.integers)

print(f"Result: {result}")