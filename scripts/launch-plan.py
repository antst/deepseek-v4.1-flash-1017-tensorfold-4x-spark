"""Render four host-local Docker argv arrays. Does not execute them."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(c, verify=False):
    profile = json.loads((ROOT/'serving/profile.json').read_text())
    provenance = json.loads((ROOT/'provenance.json').read_text())
    nodes = c['nodes']
    assert sorted(n['rank'] for n in nodes) == [0, 1, 2, 3]
    assert 1 <= c['api_port'] <= 65535 and 1 <= c['bootstrap_port'] <= 65535
    artifacts = c['artifact_paths']
    assert all(Path(v).is_absolute() for v in artifacts.values())
    expected = dict(head_binary=provenance['qualified_binaries']['head'],
                    follower_binary=provenance['qualified_binaries']['follower'],
                    local_head_kernel=provenance['qualified_extra_kernels']['local_head_pack.fatbin'],
                    prefetch_kernel=provenance['qualified_extra_kernels']['exl3_dn_prefetch.fatbin'],
                    rdma_kernel=provenance['qualified_rdma_fatbin'],
                    nccl=provenance['qualified_nccl_so'])
    if verify:
        for name,digest in expected.items():
            assert hashlib.sha256(Path(artifacts[name]).read_bytes()).hexdigest() == digest, name
    out = []
    for n in sorted(nodes, key=lambda n:n['rank']):
        rank = n['rank']; cpus = n['poster_cpus']
        assert len(cpus) == len(set(cpus)) == 6 and all(isinstance(v,int) and v >= 0 for v in cpus)
        env = dict(profile['environment'], SC_POSTER_CPUS=','.join(map(str,cpus)),
                   TF_TP_MASTER=c['rank0_address'], TF_TP_PORT=str(c['bootstrap_port']),
                   TF_RDMA_DEVICES=c['rdma_devices'],
                   TENSORFOLD_REGISTRATION_CONFIG='/opt/unused-native-registration.json')
        assert env['SC_TF_ENABLE'] == '0' and env['TF_NATIVE_COMPARE_TUNING'] == 'tuned'
        assert env['TF_DS_DRAFT_GU_COLUMNS'] == '1'
        executable = '/opt/switch-engine/'+('tensorfold-native' if rank == 0 else 'tf-dsv41-lanes')
        mounts = {artifacts['model']:'/model', artifacts['rank_inputs']:'/rank-inputs',
                  artifacts['kit']:'/kit', artifacts['engram']:'/engram',
                  artifacts['nccl']:'/opt/study-lib/libnccl.so.2',
                  artifacts['head_binary' if rank == 0 else 'follower_binary']:executable,
                  artifacts['local_head_kernel']:'/opt/recipe/local_head_pack.fatbin',
                  artifacts['prefetch_kernel']:'/opt/recipe/exl3_dn_prefetch.fatbin',
                  artifacts['rdma_kernel']:'/opt/recipe/rdma_gather.fatbin'}
        argv = ['docker','run','--name',f'ds1017-native-columns-r{rank}',
                '--network','host','--ipc','host','--gpus','all',
                '--ulimit','memlock=-1:-1','--ulimit','stack=67108864:67108864','--cap-add','IPC_LOCK']
        for dev in c['hca_devices']:argv += ['--device',dev]
        for src,dst in mounts.items():argv += ['--mount',f'type=bind,src={src},dst={dst},readonly']
        argv += ['--mount',f"type=bind,src={c['cache_root']}/rank{rank},dst=/nvme"]
        for k,v in sorted(env.items()):argv += ['--env',k+'='+v]
        image = c['head_image' if rank == 0 else 'follower_image']
        assert image and not image.startswith('REPLACE_'), 'Supply runtime image references'
        argv += ['--entrypoint',executable,image]
        if rank == 0:
            args = ['serve','/model','--backend','cuda','--host','0.0.0.0','--port',str(c['api_port']),
                    '--name',c['model_name'],'--context','1048576','--parallel','16',
                    '--max-tokens','32768','--temperature','1.0','--top-p','0.95','--top-k','20',
                    '--thinking','--prompt-cache-gib','2','--spill-dir','/nvme','--spill-gib','256',
                    '--spill-min-free-gib','64','--spill-min-tokens','8192']
        else:
            args = ['/model','/rank-inputs/rank-cache',str(rank),'4',c['rank0_address'],str(c['bootstrap_port']),
                    '/kit','--engram','/engram','--token-map','/rank-inputs/dsv41_token_map.json',
                    '--pool','10485760','--streams','16','--round-rows','48','--drafts','1',
                    '--graphs','1','--side','1','--prefetch','1','--aio','1','--context','1048576',
                    '--rdma-kernels','/opt/recipe/rdma_gather.fatbin','--rdma',c['rdma_devices']]
        out.append(dict(rank=rank,ssh_host=n['ssh_host'],argv=argv+args))
    return dict(executed=False,artifacts_hash_verified=verify,network_or_gpu_actions=False,nodes=out)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--verify-artifacts',action='store_true')
    a=ap.parse_args()
    result=render(json.loads(a.config.read_text()),a.verify_artifacts)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),executed=False,artifacts_hash_verified=a.verify_artifacts)))
