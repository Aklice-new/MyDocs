# Container
## 线性容器
### std::array
不同于std::vector，std:vector是可变容量的容器（自动扩容），但是它在删除后，申请的空间不会自动释放，需要手动去进行释放，shrink_to_fit()，释放内存。而std::array是固定大小的，所以在针对一些固定大小的数据时，优先考虑std::array。
```cpp
//初始化
// constexper int length = 4;
// length 必须是常量表达式
std::array<int, length> arr = {1, 2, 3, 4};
```
### forward_list
std::forward_list是一个列表容器，类似于std::list，但是std::list是双向链表，而std::forward_list是单向链表，提供了O(1）的插入是复杂度。注意：它不提供size()方法，当不需要双向迭代，比std::list更有空间利用率。
## 无序容器

c++11引入的 unordered_map / unordered_multimap, unordered_set / unordered_multiset 无序容器，内部实现通过hash实现。
## 元组
### std::tuple
```cpp
std::tuple<int, double, ...> t(xxxxxx); //构造函数
std::make_tuple(); // 构造元组
std::get<index>(tuple); //获取元组对应位置的值
std::tie(对应数据类型的变量...) = tuple; //将tuple中的值获取到变量中

由于std::get<>方法有问题， 所以c++17 引入了std::variant<>动态类型

#include <variant>
template <size_t n, typename... T>
constexpr std::variant<T...> _tuple_index(const std::tuple<T...>& tpl, size_t i) {
	if constexpr (n >= sizeof...(T))
	throw std::out_of_range(" 越界.");
	if (i == n)
	return std::variant<T...>{ std::in_place_index<n>, std::get<n>(tpl) };
	return _tuple_index<(n < sizeof...(T)-1 ? n+1 : 0)>(tpl, i);
}
template <typename... T>
constexpr std::variant<T...> tuple_index(const std::tuple<T...>& tpl, size_t i) {
	return _tuple_index<0>(tpl, i);
}
template <typename T0, typename ... Ts>
	std::ostream & operator<< (std::ostream & s, std::variant<T0, Ts...> const & v) {
	std::visit([&](auto && x){ s << x;}, v);
	return s;
}
通过以上的代码就可以实现很方便的访问tuple中的值
std::cout << tuple_index(t, i) << std::endl;

```

但是标准库对tuple的支持很少，使用很有限。


# 第5章  智能指针与内存管理
## RAII与引用计数

RAII（**R**esource **A**cquisition **I**s **I**nitialization），在构造的时候申请资源，在析构的时候释放资源。

### shared_ptr

shared_ptr 是一种智能指针，能记录多少个shared_ptr 共同指向了同一个对象，从而消除显式的调用delete，当计数器变为0时，会自动将对象删除。

make_shared 进行构造 c++ 11
### unique_ptr

std::unique_ptr是一种独占的智能指针， 它禁止其他智能指针与其共同指向同一个对象，保证了代码的安全性。所以在转移的时候由于独占性，不能直接复制给其他指针，而是通过std::move()进行对象的移动。

make_unique 是在c++ 14中才出现的，因为11的时候忘了。

### weak_ptr 
不太常用，有点复杂。是为了解决 shared_ptr中可能会出现的问题。

## 第6章 正则表达式

std::regex 包含在头文件库 regex 中。

### 使用方法

```cpp

// 1
std::string fnames[] = {"foo.txt", "bar.txt", "test", "a0.txt", "AAA.txt"};
std::string fnames = "aslfaj.txt"; // 待匹配字符串
std::regex txt_regex("[a-z]+\\.txt"); // 正则表达式
std::cout << std::regex_match(fname[0], txt_regex) << std::endl; // 匹配结果

// 2

std::regex base_regex("([a-z]+)\\.txt");
std::smatch base_match;
for(const auto &fname: fnames) {
	if (std::regex_match(fname, base_match, base_regex)) {
	// std::smatch 的第一个元素匹配整个字符串
	// std::smatch 的第二个元素匹配了第一个括号表达式
	if (base_match.size() == 2) {
		std::string base = base_match[1].str();
		std::cout << "sub-match[0]: " << base_match[0].str() << std::endl;
		std::cout << fname << " sub-match[1]: " << base << std::endl;
	}
}
}
```
看了[https://zhihu.com/question/23070203/answer/84248248] 这个原作者说，由于regex的实现是基于递归的，所以有可能爆栈。


## 第7章 并行与并发

### 互斥量与临界区

std::mutex 是C++11中的mutex类，可以用于创建互斥量，通过lock(),unlock()进行上锁和解锁。

C++11提供了一个RAII的类，std::lock_guard 模板类用于简化加锁和解锁的过程。

当在lock的作用域内保证互斥，但是离开