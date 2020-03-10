import datetime
import sys

def count_time(func):
    def int_time(*args, **kwargs):
        start_time = datetime.datetime.now()  # 程序开始时间
        func(*args, **kwargs)
        over_time = datetime.datetime.now()   # 程序结束时间
        total_time = (over_time-start_time).total_seconds()
        print('其执行时间:%s秒' % total_time)
    return int_time


@count_time
def run():
    print("当前方法是:", sys._getframe().f_code.co_name)
    sum = 0
    for i in range(1,10000):
        sum = sum + i


if __name__ == '__main__':
    run()