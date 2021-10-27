#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#define _CRT_SECURE_NO_WARNINGS     // localtime/fopen���뾯��

// ��־����
// ʹ�÷�ʽ��LOGC("INFO", "a=%d", 10); LOGC("ERR", "initial model failed");LOG("INFO", "got w=%d,h=%d,c=%d", 1, 2, 3);
// ��־��Ϣ�Զ����浽��ǰĿ¼�µ�logfile.log�ļ���
inline void logC(const char* func, const char* file, const int line,
    const char* type, const char* format, ...)
{
    FILE* file_fp;
    time_t loacl_time;
    char time_str[128];

    // ��ȡ����ʱ��
    time(&loacl_time);
    strftime(time_str, sizeof(time_str), "[%Y.%m.%d %X]", localtime(&loacl_time));

    // ��־���ݸ�ʽת��
    va_list ap;
    va_start(ap, format);
    char fmt_str[2048];
    vsnprintf(fmt_str, sizeof(fmt_str), format, ap);
    va_end(ap);

    // ����־�ļ�
    file_fp = fopen("./logfile.log", "a");

    // д�뵽��־�ļ���
    if (file_fp != NULL)
    {
        fprintf(file_fp, "[%s]%s[%s@%s:%d] %s\n", type, time_str, func,
            file, line, fmt_str);
        fclose(file_fp);
    }
    else
    {
        fprintf(stderr, "[%s]%s[%s@%s:%d] %s\n", type, time_str, func,
            file, line, fmt_str);
    }
}

#define LOGC(type, format, ...) logC(__func__, __FILE__, __LINE__, type, format, ##__VA_ARGS__)