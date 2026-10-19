import socket


def main():
    #  1.创建udp套接字
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 可选(接收时的端口 ('',端口))
    local_address = ("", PORT)
    udp_socket.bind(local_address)

    while True:
        # 2.发送数据到指定的电脑
        send_data = input("请输入你要发送的数据:")
        udp_socket.sendto(send_data.encode("utf-8"), ("", PORT))
        if send_data == "exit":
            break

        # 可选(接收对方发送的数据)
        recv_data = udp_socket.recvfrom(1024)  # 1024表示本次接收的最大字节数
        recv_msg = recv_data[0].decode("utf-8")
        recv_address = recv_data[1]
        print("当前客户端地址：{}，当前接受的信息是：{}".format(recv_address, recv_msg))

    # 3.关闭套接字
    udp_socket.close()


if __name__ == "__main__":
    main()
