import os
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import time


class RemoteHost:
    addr: str
    keyfile_path: str
    login_name: str
    container_id: Optional[str] = None
    ami: Optional[str] = None

    def __init__(self, addr: str, keyfile_path: str, login_name: str = 'ubuntu'):
        self.addr = addr
        self.keyfile_path = keyfile_path
        self.login_name = login_name

    def _gen_ssh_prefix(self) -> List[str]:
        return ["ssh", "-o", "StrictHostKeyChecking=no", "-i", self.keyfile_path,
                f"{self.login_name}@{self.addr}", "--"]

    @staticmethod
    def _split_cmd(args: Union[str, List[str]]) -> List[str]:
        return args.split() if isinstance(args, str) else args

    def run_ssh_cmd(self, args: Union[str, List[str]]) -> None:
        subprocess.check_call(self._gen_ssh_prefix() + self._split_cmd(args))

    def check_ssh_output(self, args: Union[str, List[str]]) -> str:
        return subprocess.check_output(self._gen_ssh_prefix() + self._split_cmd(args)).decode("utf-8")

    def scp_upload_file(self, local_file: str, remote_file: str) -> None:
        subprocess.check_call(["scp", "-i", self.keyfile_path, local_file,
                              f"{self.login_name}@{self.addr}:{remote_file}"])

    def scp_download_file(self, remote_file: str, local_file: Optional[str] = None) -> None:
        if local_file is None:
            local_file = "."
        subprocess.check_call(["scp", "-i", self.keyfile_path,
                              f"{self.login_name}@{self.addr}:{remote_file}", local_file])

    def start_docker(self, image="quay.io/pypa/manylinux2014_aarch64:latest") -> None:
        self.run_ssh_cmd("sudo apt-get install -y docker.io")
        self.run_ssh_cmd(f"sudo usermod -a -G docker {self.login_name}")
        self.run_ssh_cmd("sudo service docker start")
        self.run_ssh_cmd(f"docker pull {image}")
        self.container_id = self.check_ssh_output(f"docker run -t -d -w /root {image}").strip()

    def using_docker(self) -> bool:
        return self.container_id is not None

    def run_cmd(self, args: Union[str, List[str]]) -> None:
        if not self.using_docker():
            return self.run_ssh_cmd(args)
        assert self.container_id is not None
        docker_cmd = self._gen_ssh_prefix() + ['docker', 'exec', '-i', self.container_id, 'bash -i ']
        p = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE)
        p.communicate(input=" ".join(["source ~/.bashrc;"] + self._split_cmd(args)).encode("utf-8"))
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, docker_cmd)

    def check_output(self, args: Union[str, List[str]]) -> str:
        if not self.using_docker():
            return self.check_ssh_output(args)
        assert self.container_id is not None
        docker_cmd = self._gen_ssh_prefix() + ['docker', 'exec', '-i', self.container_id, 'bash -i ']
        p = subprocess.Popen(docker_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        (out, err) = p.communicate(input=" ".join(["source ~/.bashrc;"] + self._split_cmd(args)).encode("utf-8"))
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, docker_cmd, output=out, stderr=err)
        return out.decode("utf-8")

    def upload_file(self, local_file: str, remote_file: str) -> None:
        if not self.using_docker():
            return self.scp_upload_file(local_file, remote_file)
        tmp_file = os.path.join("/tmp", os.path.basename(local_file))
        self.scp_upload_file(local_file, tmp_file)
        self.run_ssh_cmd(["docker", "cp", tmp_file, f"{self.container_id}:/root/{remote_file}"])
        self.run_ssh_cmd(["rm", tmp_file])

    def download_file(self, remote_file: str, local_file: Optional[str] = None) -> None:
        if not self.using_docker():
            return self.scp_download_file(remote_file, local_file)
        tmp_file = os.path.join("/tmp", os.path.basename(remote_file))
        self.run_ssh_cmd(["docker", "cp", f"{self.container_id}:/root/{remote_file}", tmp_file])
        self.scp_download_file(tmp_file, local_file)
        self.run_ssh_cmd(["rm", tmp_file])

    def download_wheel(self, remote_file: str, local_file: Optional[str] = None) -> None:
        if self.using_docker() and local_file is None:
            basename = os.path.basename(remote_file)
            local_file = basename.replace("-linux_aarch64.whl", "-manylinux2014_aarch64.whl")
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