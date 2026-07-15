# 1:5000 第一阶段临时基线

当前只登记服务器训练日志中的临时最佳结果。`checkpoint_best_epoch5.pt` 需要从服务器上传后放入本目录；checkpoint 文件被 Git 忽略。

- epoch: 5
- 洛杉矶完整验证集 greedy exact: 0.26282932283537935
- teacher-forced token accuracy: 0.875145019365663
- epoch 8 起出现 NaN，epoch 8 及之后 checkpoint 无效
- 当前结果未完成复杂建筑分组验收，也不代表 1:10000 已完成
