import os
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import time
from . import ec2_ops


class RemoteHost:
    addr: str
    keyfile_path: str
    login_name: str
    container_id: Optional[str] = None
    ami: Optional[str] = None
    sg_id: str = None
    instance = None
    keep_instance: bool = False
    local_dir: str = None

    def __init__(self, addr: str, keyfile_path: str, login_name: str = "ubuntu"):
        self.addr = addr
        self.keyfile_path = keyfile_path
        self.login_name = login_name

    def _gen_ssh_prefix(self) -> List[str]:
        return [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "TCPKeepAlive=yes",
            "-o",
            "ServerAliveInterval=30",
            "-i",
            self.keyfile_path,
            f"{self.login_name}@{self.addr}",
            "--",
        ]

    @staticmethod
    def _split_cmd(args: Union[str, List[str]]) -> List[str]:
        return args.split() if isinstance(args, str) else args

    def run_ssh_cmd(self, args: Union[str, List[str]]) -> None:
        try:
            subprocess.check_call(self._gen_ssh_prefix() + self._split_cmd(args))
        except Exception as x:
            print("SSH Host Command Failed...\n"
                  f"error: {x}")
            if self.keep_instance:
                exit(1)
            ec2_ops.cleanup(self.instance, self.sg_id)
            exit(1)  

    def check_ssh_output(self, args: Union[str, List[str]]) -> str:
        return subprocess.check_output(
            self._gen_ssh_prefix() + self._split_cmd(args)
        ).decode("utf-8")

    def scp_upload_file(self, local_file: str, remote_file: str) -> None:
        try:
            subprocess.check_call(
                [
                    "scp",
                    "-i",
                    self.keyfile_path,
                    local_file,
                    f"{self.login_name}@{self.addr}:{remote_file}",
                ]
            )
        except Exception as x:
            print("Upload Failed...\n"
                  f"error: {x}")
            if self.keep_instance:
                exit(1)
            ec2_ops.cleanup(self.instance, self.sg_id)
            exit(1)     

    def scp_download_file(
        self, remote_file: str, local_file: Optional[str] = None
    ) -> None:
        rel_path = self.local_dir + "/" if self.local_dir else ""
        if local_file is None:
            local_file = "."
        try:
            subprocess.check_call(
                [
                    "scp",
                    "-i",
                    self.keyfile_path,
                    f"{self.login_name}@{self.addr}:{remote_file}",
                    rel_path + local_file,
                ]
            )
        except Exception as x:
            print("Download Failed...\n"
                  f"error: {x}")
            if self.keep_instance:
                exit(1)
            ec2_ops.cleanup(self.instance, self.sg_id)
            exit(1)       

    def start_docker(self, image: str = None, enable_cuda: bool = False) -> None:
        print("Installing Docker...")
        try:
            # installing docker.io, but insuring no daemon.json as it seems the package is
            # installing it ang setting the bridge to none, breaking container networking.
            self.run_ssh_cmd(
                "sudo apt-get install -y docker.io; "
                "sudo rm /etc/docker/daemon.json; "
                f"sudo usermod -a -G docker {self.login_name}; "
                "sudo systemctl restart docker"
            )
            if enable_cuda:
                print("Configuring Docker for nVidia GPU usage..")
                self.run_ssh_cmd(
                    "distribution=$(. /etc/os-release;echo $ID$VERSION_ID) "
                    "&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
                    "sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg "
                    "&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | "
                    "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
                    "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
                )
                self.run_ssh_cmd(
                    "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
                )
                self.run_ssh_cmd("sudo nvidia-ctk runtime configure --runtime=docker")
                self.run_ssh_cmd("sudo systemctl restart docker")
                cmd_str = f"docker run -td --gpus all -w /root {image}"
            else:
                cmd_str = f"docker run -td -w /root {image}"

            self.run_ssh_cmd(f"docker pull {image}")
            self.container_id = self.check_ssh_output(cmd_str).strip()
        except Exception as x:
            print("failed to start docker container")
            print(x)
            ec2_ops.cleanup(self.instance, self.sg_id)
            exit(1)

    def using_docker(self) -> bool:
        return self.container_id is not None

    def run_cmd(self, args: Union[str, List[str]], allow_error: bool = False) -> None:
        if not self.using_docker():
            return self.run_ssh_cmd(args)
        assert self.container_id is not None
        docker_cmd = self._gen_ssh_prefix() + [
            "docker",
            "exec",
            "-i",
            self.container_id,
            "bash -i ",
        ]
        p = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE)
        p.communicate(
            input=" ".join(["source ~/.bashrc;"] + self._split_cmd(args)).encode(
                "utf-8"
            )
        )
        rc = p.wait()
        try:
            if rc != 0:
                raise subprocess.CalledProcessError(rc, docker_cmd)
        except subprocess.CalledProcessError as x:
            if allow_error:
                return
            print("Command Execution Failed...")
            print(docker_cmd)
            if self.keep_instance:
                exit(1)
            ec2_ops.cleanup(self.instance, self.sg_id)
            exit(1)

    def check_output(self, args: Union[str, List[str]]) -> str:
        if not self.using_docker():
            return self.check_ssh_output(args)
        assert self.container_id is not None
        docker_cmd = self._gen_ssh_prefix() + [
            "docker",
            "exec",
            "-i",
            self.container_id,
            "bash -i ",
        ]
        p = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        (out, err) = p.communicate(
            input=" ".join(["source ~/.bashrc;"] + self._split_cmd(args)).encode(
                "utf-8"
            )
        )
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, docker_cmd, output=out, stderr=err)
        return out.decode("utf-8")

    def upload_file(self, local_file: str, remote_file: str) -> None:
        if not self.using_docker():
            return self.scp_upload_file(local_file, remote_file)
        tmp_file = os.path.join("/tmp", os.path.basename(local_file))
        self.scp_upload_file(local_file, tmp_file)
        self.run_ssh_cmd(
            ["docker", "cp", tmp_file, f"{self.container_id}:/root/{remote_file}"]
        )
        self.run_ssh_cmd(["rm", tmp_file])

    def download_file(self, remote_file: str, local_file: Optional[str] = None) -> None:
        if not self.using_docker():
            return self.scp_download_file(remote_file, local_file)
        tmp_file = os.path.join("/tmp", os.path.basename(remote_file))
        self.run_ssh_cmd(
            ["docker", "cp", f"{self.container_id}:/root/{remote_file}", tmp_file]
        )
        self.scp_download_file(tmp_file, local_file)
        self.run_ssh_cmd(["rm", tmp_file])

    def download_wheel(
        self, remote_file: str, local_file: Optional[str] = None
    ) -> None:
        if self.using_docker() and local_file is None:
            local_file = os.path.basename(remote_file)
        self.download_file(remote_file, local_file)

    def list_dir(self, path: str) -> List[str]:
        return self.check_output(["ls", "-1", path]).split("\n")


def wait_for_connection(addr, port, timeout=15, attempt_cnt=5):
    import socket

    for i in range(attempt_cnt):
        try:
            with socket.create_connection((addr, port), timeout=timeout):
                return
        except (ConnectionRefusedError, socket.timeout):
            if i == attempt_cnt - 1:
                raise
            time.sleep(timeout)
