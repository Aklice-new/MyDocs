# Caffe中的Protobuf

        caffe是采用了[Protobuf]([Overview | Protocol Buffers Documentation](https://protobuf.dev/overview/))这种文件格式定义网络的结构，通过.prototxt来定义网络中的一些参数，如网络的名称、各个layer的相关属性等等一些参数的数据结构。

        其中src/caffe/proto/caffe.proto文件中就是关于Protobuf相关数据结构的定义说明，在编译过程中，会根据该文件生成 /build/include/caffe/proto/caffe.pb.cc 和 caffe.pb.h。这两个文件中包含了C++的Class对应Protobuf的数据结构。

## Protobuf

Protocol Buffers 是一种与语言无关、与平台无关的可扩展机制，用于序列化结构化数据。

使用协议缓冲区的一些优点包括： 

-    紧凑的数据存储 

-    快速解析 

-    可用于多种编程语言 

-    通过自动生成的类优化功能

<img title="" src="file:///home/aklice/文档/MyDocs/Caffe/imgs/protobuf_workflow.png" alt="protobuf_workflow.png" data-align="center">



## Protobuf的加载

其中所有prototxt中定义的数据结构都是通过Protobuf自动生成对应的数据结构到caffe.pb.cc和caffe.pb.h中，生成后的这些数据结构类都是Protobuf中的数据结构的子类，通过调用父类的读入和生成的类的parsing方法将对应的字段进行读入。

## Protobuf的数据定义方式

Protobuf的数据结构定义文件以.proto结尾，每一个数据结构都message + 类名的方式命名，然后在大括号中对成员变量进行声明。所有的成员变量都应该有一个标签：

1. optional：一个optional修饰的成员变量有两种可能：
   
   1. 该字段已设置，并且包含显式设置或从线路解析的值。它将被序列化。
   
   2.  该字段未设置，将返回默认值。它不会被序列化

2. repeated： 该字段类型可以在格式正确的消息中重复零次或多次。重复值的顺序将被保留。

3. map: 这是成对的键/值字段类型。有关此字段类型的更多信息，请参阅[[Encoding | Protocol Buffers Documentation](https://protobuf.dev/programming-guides/encoding/#maps)]。

4. 如果没有应用显式字段标签，则假定使用默认字段标签，称为“隐式字段存在”

在proto文件中可以定义嵌套类型，但是会导致数据结构臃肿，同时带来依赖关系复杂。

不同编程语言下各个数据类型对应的具体名字[Language Guide (proto 3) | Protocol Buffers Documentation](https://protobuf.dev/programming-guides/proto3/#scalar)

```cpp
syntax = "proto3";

package tutorial; // 避免不同的命名空间之间的冲突

message Person {
  optional string name = 1;
  optional int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    PHONE_TYPE_UNSPECIFIED = 0;
    PHONE_TYPE_MOBILE = 1;
    PHONE_TYPE_HOME = 2;
    PHONE_TYPE_WORK = 3;
  }

  message PhoneNumber {
    optional string number = 1;
    optional PhoneType type = 2 [default = PHONE_TYPE_HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

关于Protobuf的细节使用还请参考[Protocol Buffers 官网](https://protobuf.dev/)



## 对比

Protocal Buffer说到底也只是一种用于数据序列化的工具，对比其他序列化的工具如：json等有什么差异呢？

[序列化和反序列化 - 美团技术团队](https://tech.meituan.com/2015/02/26/serialization-vs-deserialization.html)]


