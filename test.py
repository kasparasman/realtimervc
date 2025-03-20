import torch
# #  Assume model_dict is the dictionary loaded from torch.load() of your model checkpoint.

def verify_checkpoint(path):
    try:
        cpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print("Error loading checkpoint:", e)
        return

    # Check 'config'
    if "config" not in cpt:
        print("Missing key: 'config'")
    else:
        print("config exists. Target sample rate (last element):", cpt["config"][-1])
    
    # Check 'weight' and 'emb_g.weight'
    if "weight" not in cpt:
        print("Missing key: 'weight'")
    else:
        if "emb_g.weight" not in cpt["weight"]:
            print("Missing key: 'emb_g.weight' inside 'weight'")
        else:
            print("emb_g.weight shape:", cpt["weight"]["emb_g.weight"].shape)
    
    # Check for F0 flag and version
    f0_val = cpt.get("f0", None)
    version_val = cpt.get("version", None)
    print("F0 flag:", f0_val if f0_val is not None else "default (1)")
    print("Version:", version_val if version_val is not None else "default ('v1')")

# Replace with your actual model path:
verify_checkpoint("C:/Users/Kasparas/realtimervc/assets/weights/argos.pth")
