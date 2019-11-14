## 安装说明
1. 环境
    1. CentOS Linux release 7.6.1810
    2. Python==3.7.4
    3. virtualenv==16.7.2
    4. mongodb==4.0.1
    
2. 激活虚拟环境
    1. source py37env/bin/activate(虚拟环境路径)
3. 安装第三方包
    1. pip install -r requirements.txt
    2. pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
    3. pip3 install torchvision
4. mongodb数据库示例
    ![Image text](data_exp.png)
    
## 配置说明
```
Config.ini
```

## 程序启动
```
cd ProjectPath
python manage.py > log.txt 2>&1 &
```





