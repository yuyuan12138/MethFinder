import torch


def save_model(state_dict, path_name):
    print("<===== Saving model =====>")
    torch.save(state_dict, path_name)
    print("<===== Saved =====>")

def load_model(model_path, map_location, ):
    print("<===== Loading model =====>")
    model = torch.load(model_path, map_location)
    print("<===== Loaded =====>")
    return model
    