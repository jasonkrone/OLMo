import huggingface_hub
from transformers import AutoModelForCausalLM
from hf_olmo.convert_olmo_to_hf import download_remote_checkpoint_and_convert_to_hf

KEYS_PATH = "./keys.env"

CKPT_LIST = [
    #{
    #    "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-small/w1r5xfzt/step5000-unsharded/",
    #    "local": "/artifacts/olmo_1b_toks_21b",
    #    "id": "jasonkrone/olmo_1b_toks_21b",
    #},
    #{
    #    "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-small/s7wptaol/step12000-unsharded/",
    #    "local": "/artifacts/olmo_1b_toks_50b",
    #    "id": "jasonkrone/olmo_1b_toks_50b",
    #},
    {
        "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-small/s7wptaol/step18000-unsharded/",
        "local": "/artifacts/olmo_1b_toks_75b",
        "id": "jasonkrone/olmo_1b_toks_75b",
    },
    {
        "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-small/sw58clgr/step30000-unsharded/",
        "local": "/artifacts/olmo_1b_toks_126",
        "id": "jasonkrone/olmo_1b_toks_126",
    },
    {
        "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-medium/l6v218f4/step40000-unsharded/",
        "local": "/artifacts/olmo_7b_toks_168b",
        "id": "jasonkrone/olmo_7b_toks_168b",
    },
    {
        "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-medium/mk9kaqh0/step72000-unsharded/",
        "local": "/artifacts/olmo_7b_toks_302b",
        "id": "jasonkrone/olmo_7b_toks_302b",
    },
    {
        "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-medium/hrshlkzq/step107000-unsharded/",
        "local": "/artifacts/olmo_7b_toks_449b",
        "id": "jasonkrone/olmo_7b_toks_449b",
    },
    {
        "remote": "https://olmo-checkpoints.org/ai2-llm/olmo-medium/eysi0t0y/step143000-unsharded/",
        "local": "/artifacts/olmo_7b_toks_600b",
        "id": "jasonkrone/olmo_7b_toks_600b",
    },
]


def read_token(key_path, key_name):
    with open(key_path, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if key_name in l]
    assert len(lines) == 1
    token = lines[0].split("=")[-1].strip().replace('"', "")
    return token


def main():
    huggingface_hub.login(token=read_token(KEYS_PATH, "HUGGINFGACE_WRITE_TOKEN"))
    for ckpt_dict in CKPT_LIST:
        model_id = ckpt_dict["id"]
        print(f"===================== Downloading {model_id} ================================\n\n")
        download_remote_checkpoint_and_convert_to_hf(ckpt_dict["remote"], ckpt_dict["local"])
        hf_model = AutoModelForCausalLM.from_pretrained(ckpt_dict["local"])
        hf_model.push_to_hub(repo_id=model_id)
        print(f"=============================================================================\n\n")
        del hf_model


if __name__ == "__main__":
    main()






