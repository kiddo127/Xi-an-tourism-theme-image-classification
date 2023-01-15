# Xi-an-tourism-theme-image-classification

使用 `inference.sh` 执行推理任务，`inference.sh` 的内容如下：

```sh
python inference.py --test-path test_images \
# --attention
```

`--test-path` 为测试图片文件夹路径，默认为 `test_images`

`--attention` 为是否加入 attention 机制，默认不使用 attention 机制，将第二行取消注释后，可以加入 attention 机制



将训练好的模型权重放入 `weights` 文件夹下，加入 attention 机制的模型权重命名为 Att_4layers_best.ckpt ，不加入 attention 机制的模型权重命名为 DA_resnet50_best.ckpt 模型文件下载地址：[kiddo127的模型-昇思大模型平台 (mindspore.cn)](https://xihe.mindspore.cn/models/kiddo127/att_resnet/tree)

将待测试图片放入指定文件夹下，或修改 `inference.sh` 文件中的 `--test-path` 参数为待测试图片文件夹路径



执行 `sh inference.sh` ，开始推理



