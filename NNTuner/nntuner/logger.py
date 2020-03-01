# coding:utf-8
import os
import logging
from logging import handlers


def get_logger(log_filename, level=logging.DEBUG, when='midnight', back_count=0):
    """
    :brief  日志记录
    :param log_filename: 日志名称
    :param level: 日志等级
    :param when: 间隔时间:
        S:秒
        M:分
        H:小时
        D:天
        W:每星期（interval==0时代表星期一）
        midnight: 每天凌晨
    :param back_count: 备份文件的个数，若超过该值，就会自动删除
    :return: logger
    """
    logger = logging.getLogger(log_filename)
    logger.setLevel(level)
    log_path = os.path.join(LOG_ROOT, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, log_filename)
    # log输出格式
    formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # 输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # 输出到文件
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        when=when,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    # 设置日志输出格式
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 添加到logger对象里
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    logger = logging.getLogger("autotvm")
    logger.setLevel(logging.DEBUG)
    logger.debug("debug test")
    logger.info("info test")
    logger.warn("warn test")
    logger.error("error test")