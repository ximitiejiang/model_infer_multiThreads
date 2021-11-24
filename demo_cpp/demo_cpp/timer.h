#pragma once
#include <chrono>
#include <iostream>
#include <string>

// 高精度计时器
class Timer
{
	typedef std::chrono::steady_clock::time_point time_point_type;
public:
	// 启动计时
	void start() { steady_start = std::chrono::steady_clock::now(); }
	// 终止计时
	void stop() { steady_end = std::chrono::steady_clock::now(); }
	// 返回时间(ms)
	double stop_and_return() { stop(); return get_time() * 1000; }
	// 终止计时并输出结果
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
