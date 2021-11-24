#pragma once
#include <chrono>
#include <iostream>
#include <string>

// �߾��ȼ�ʱ��
class Timer
{
	typedef std::chrono::steady_clock::time_point time_point_type;
public:
	// ������ʱ
	void start() { steady_start = std::chrono::steady_clock::now(); }
	// ��ֹ��ʱ
	void stop() { steady_end = std::chrono::steady_clock::now(); }
	// ����ʱ��(ms)
	double stop_and_return() { stop(); return get_time() * 1000; }
	// ��ֹ��ʱ��������
	void stop_and_show(std::string name)
	{
		stop();
		std::cout << name << get_time() * 1000 << " msec" << std::endl;
	}
private:
	time_point_type steady_start, steady_end;
	double get_time()
	{
		double took_time = std::chrono::duration<double>(steady_end - steady_start).count();
		return took_time;
	}
};
