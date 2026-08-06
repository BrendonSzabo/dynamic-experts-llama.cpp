#!/usr/bin/env python3

import os
import ctypes
import subprocess
import sys
from pathlib import Path

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
RESET = "\033[0m"

def check(label):
    print(f"  {label}... ", end="", flush=True)

def pass_(msg=""):
    print(f"{GREEN}PASS{'' if not msg else ' — ' + msg}{RESET}")
    return True

def fail(msg=""):
    print(f"{RED}FAIL{'' if not msg else ' — ' + msg}{RESET}")
    return False

def warn(msg=""):
    print(f"{YELLOW}WARN{'' if not msg else ' — ' + msg}{RESET}")
    return True


# ── 1. GPU present ──

def check_gpu():
    check("GPU present")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=5)
        if "GPU" not in out:
            return fail("no GPU found")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return fail("nvidia-smi not found or timed out")
    return pass_(out.strip().split("\n")[0] if "\n" in out else out.strip())


# ── 2. nvidia-fs kernel module ──

def check_nvidia_fs():
    check("nvidia-fs kernel module")
    try:
        out = subprocess.check_output(["lsmod"], text=True, timeout=3)
    except:
        return fail("lsmod failed")
    if "nvidia_fs" in out:
        return pass_()
    return fail("not loaded — sudo modprobe nvidia-fs or check dkms status")


# ── 3. cuFile driver ──

def check_cufile():
    check("cuFile driver")
    try:
        lib = ctypes.CDLL("libcufile.so")
    except OSError as e:
        return fail(str(e))

    lib.cuFileDriverOpen.restype = ctypes.c_int
    lib.cuFileDriverClose.restype = ctypes.c_int

    CU_FILE_SUCCESS = 0
    rc = lib.cuFileDriverOpen()
    if rc != CU_FILE_SUCCESS:
        return fail(f"cuFileDriverOpen returned {rc}")
    lib.cuFileDriverClose()
    return pass_()


# ── 4. Filesystem O_DIRECT ──

def check_direct(path):
    check(f"O_DIRECT on filesystem")

    import tempfile
    dirname = str(Path(path).parent)
    tmp = ""
    try:
        fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".gds_test_")
        os.write(fd, b"\x00" * 4096)
        os.fsync(fd)
        os.close(fd)
        fd = os.open(tmp, os.O_RDONLY | os.O_DIRECT)
        os.close(fd)
        os.unlink(tmp)
    except OSError as e:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        return fail(str(e))
    return pass_()


# ── 5. cuFile file registration ──

CUFILE_ERR_MSGS = {
    4000: "CU_FILE_NOT_OPENED — cuFileDriverOpen not called",
    4001: "CU_FILE_INVALID — file handle invalid",
    4002: "CU_FILE_ALREADY_OPEN",
    4003: "CU_FILE_CLOSED",
    5000: "CU_FILE_DEVICE_UNSUPPORTED — filesystem type not GDS-compatible",
    5001: "CU_FILE_IOCTL_FAILED — cannot access device (permissions?)",
    5002: "CU_FILE_INTERNAL_ERROR",
    5005: "CU_FILE_REGISTRATION_FAILED — cannot pin/map file for DMA",
    5008: "CU_FILE_OPEN_FAILED — nvidia-fs module not loaded, or file not GDS-compatible",
    5010: "CU_FILE_CLOSE_FAILED",
}

def check_cufile_register(path):
    check(f"cuFile register file")

    try:
        lib = ctypes.CDLL("libcufile.so")
    except OSError as e:
        return fail(str(e))

    class CUfileDescr(ctypes.Structure):
        _fields_ = [
            ("type",   ctypes.c_int),
            ("cookie", ctypes.c_int32 * 8),
            ("fd",     ctypes.c_int),
            ("handle", ctypes.c_void_p),
        ]

    lib.cuFileHandleRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUfileDescr)]
    lib.cuFileHandleRegister.restype  = ctypes.c_int

    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    desc = CUfileDescr()
    desc.type = 0
    desc.fd   = fd

    fh = ctypes.c_void_p()
    rc = lib.cuFileHandleRegister(ctypes.byref(fh), ctypes.byref(desc))
    os.close(fd)

    if rc != 0:
        msg = CUFILE_ERR_MSGS.get(rc, f"error {rc}")
        return fail(msg)

    lib.cuFileHandleDeregister.argtypes = [ctypes.c_void_p]
    lib.cuFileHandleDeregister.restype  = ctypes.c_void_p
    lib.cuFileHandleDeregister(fh)
    return pass_()


# ── 6. GPU↔NVMe topology ──

def check_topo():
    check("GPU↔NVMe affinity")

    try:
        out = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, timeout=5)
    except:
        return warn("nvidia-smi topo failed — check manually")

    lines = out.strip().split("\n")

    nvme_rows = []
    for i, line in enumerate(lines):
        if "NVMe" in line:
            nvme_rows.append((i, line))

    if not nvme_rows:
        return warn("no NVMe drives visible in GPU topology — "
                    "NVMe may be behind a non-PCIe controller or SATA bridge; "
                    "GDS DMA path likely unavailable")

    gpu_col_names = []
    for i, line in enumerate(lines):
        if i > 5:
            break
        parts = line.split()
        for p in parts:
            if p.startswith("GPU"):
                gpu_col_names.append(p)

    for _, row in enumerate(nvme_rows):
        _, line = row
        parts = line.split()
        drive = parts[0] if parts else "NVMe"
        for idx, gpu_name in enumerate(gpu_col_names):
            col = idx + 1
            if col < len(parts):
                affinity = parts[col]
                if affinity in ("PIX", "PXB"):
                    return pass_(f"{drive} ↔ {gpu_name} = {affinity}")
                if affinity in ("PHB", "NODE", "SYS"):
                    return warn(f"{drive} ↔ {gpu_name} = {affinity} — "
                                "crosses PCIe bridges, GDS may bounce through CPU")

    return warn("NVMe present but not on same PCIe switch as GPU — GDS unavailable")


# ── main ──

def main():
    if len(sys.argv) < 2:
        print("usage: test_gds.py <path-to-expert-bin>")
        sys.exit(1)

    bin_path = sys.argv[1]
    if not os.path.isfile(bin_path):
        print(f"file not found: {bin_path}")
        sys.exit(1)

    print("\nGDS availability check\n")

    ok  = check_gpu()
    ok &= check_nvidia_fs()
    ok &= check_cufile()
    ok &= check_direct(bin_path)
    ok &= check_cufile_register(bin_path)
    ok &= check_topo()

    print(f"\n{'ALL CHECKS PASSED — GDS available' if ok else 'GDS NOT AVAILABLE — see failures above'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
