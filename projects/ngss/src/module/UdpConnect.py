from paho.mqtt.client import *
import paho.mqtt.client as mqtt
import yaml
import hashlib
import threading
import os
import json
import base64



class MQTTPublisher:
    def __init__(self):
        """
        初始化 MQTT 发布者。
        
        :param broker: MQTT 代理服务器地址
        :param port: MQTT 代理服务器端口
        :param username: MQTT 用户名
        :param password: MQTT 密码
        
        这里的函数在开源之前需要删除，需要告知用户让他们自己配网。
        上面的内容也需要被删除，所有有关兴隆网络配置的内容不要放出来。 
        """


    def on_connect(self, client, userdata, flags, rc):
        """
        连接回调函数。
        
        :param client: 客户端实例
        :param userdata: 用户数据
        :param flags: 响应标志
        :param rc: 连接返回码
        """
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_publish(self, client, userdata, mid):
        """
        消息发布回调函数。
        
        :param client: 客户端实例
        :param userdata: 用户数据
        :param mid: 消息ID
        """
        print(f"Message published with mid {mid}")
        
    def on_message(self, client, userdata, msg):
        message = msg.payload.decode()
        print(f"Received message '{message}' on topic '{msg.topic}'")
        if message == "Receive failed":
            self.republish()
        elif message == "Receive success":
            self.success_received.set()  # 设置事件标志

    def connect(self):
        """
        连接到 MQTT 代理。
        """
        self.client.username_pw_set(self.username, self.password)
        self.client.connect(self.broker, self.port)
        self.client.loop_start()
        for topic, _ in topics:
            self.client.subscribe(topic)

    def publish(self, topic, payload):
        """
        发布消息到主题。
        
        :param topic: 要发布的消息主题
        :param payload: 要发布的消息内容
        """
        self.client.publish(topic, payload)

    def disconnect(self):
        """
        断开与 MQTT 代理的连接。
        """
        self.client.loop_stop()
        self.client.disconnect()

    def load_topics(self, yaml_file):
        """
        从 YAML 文件中加载主题配置。
        
        :param yaml_file: YAML 文件路径
        """
        os.chdir("/home/pod/shared-nvme/NGSS/")
        with open(yaml_file, 'r') as file:
            self.topics_config = yaml.safe_load(file)

    def republish(self):
        if self.last_payload and self.last_topic:
            self.publish(self.last_topic, self.last_payload)
            print(f"Republishing payload to topic {self.last_topic}")

    def publish_to_telescope(self, section, location, telescope, schedule):
        """
        根据 YAML 文件中的配置发布消息到指定的望远镜主题。
        
        :param section: YAML 文件中的部分（例如 'ftp_transfer' 或 'nina_action'）
        :param location: 地理位置（例如 'xinglong'）
        :param telescope: 望远镜编号（例如 '1'）
        :param payload: 要发布的消息内容
        """
        telescope_name = 'telescope'+telescope
        print(telescope_name)
        if section == "nina_action":
            if section in self.topics_config and location in self.topics_config[section] and telescope_name in self.topics_config[section][location]:
                topic = self.topics_config[section][location][telescope_name]['topic']
                self.publish(topic, schedule)
                
            else:
                print("Topic not found in configuration.")
        elif section == "ftp_transfer":         
            if section in self.topics_config and location in self.topics_config[section] and telescope_name in self.topics_config[section][location]:
                topic = self.topics_config[section][location][telescope_name]['topic']
                # 计算schedule的哈希值
                schedule_hash = hashlib.sha256(schedule).hexdigest()

                # 将哈希值和schedule一起打包成payload
                payload_dict = {
                    'schedule': base64.b64encode(schedule).decode('utf-8'),
                    'hash': schedule_hash
                }
                print(schedule)
                payload = json.dumps(payload_dict)
                self.last_payload = payload
                self.last_topic = topic
                print(topic)
                self.publish(topic, payload)
            else:
                print("Topic not found in configuration.")

