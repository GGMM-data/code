## 上面几个改的比较少
maddpg-tmc:maddpg上改的，发表在tmc上
maddpg-tmc-optimize:相对于maddpg-tmc，在obs上改了一点点，提高训练速度，效果没有改变
maddpg-tmc-transfer:相对于maddpg-tmc，在obs上改了一点点，同时使用distral框架，实现迁移学习，buffer是自己实现的，actor和critic分开
maddpg-tmc-buffer-transfer:和maddpg-tmc-transfer的区别在于，这里的buffer在上面的基础上做了一点改进
maddpg-tmc-total-transfer:相对于maddpg-tmc，在obs上改了一点点，同时使用distral框架，实现迁移学习，buffer是原文的，acotr和critic在一起
**这个**maddpg-tmc-seperate-transfer:相对于maddpg-tmc，在obs上改了一点点，同时使用distral框架，实现迁移学习，buffer是原文的，acotr和critic分开

## 下面几个改的比较多
maddpg-transfer:为了提高训练速度，改了许多其他地方，没有使用lstm，同时使用distral框架，性能有所下降，buffer是改了的。
maddpg-lstm-transfer:为了提高训练速度，改了许多其他地方，使用lstm，同时使用distral框架，性能有所下降
maddpg-lstm:为了提高训练速度，改了许多其他地方，使用lstm，同时使用distral框架，但是只有一个任务性能有所下降

