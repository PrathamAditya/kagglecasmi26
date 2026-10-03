import sys, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("==", p)
    for k, v in ck.items():
        if k == "model":
            print("  model: state_dict,", len(v), "tensors")
        else:
            print(" ", k, repr(v)[:400])
