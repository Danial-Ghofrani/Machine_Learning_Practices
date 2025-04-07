## loading model
import pickle

with open("model.dat", "rb") as file:
    model = pickle.load(file)

print(model.predict(([10, 168, 74, 0, 0, 38, 0.537, 34])))

# if we want to train a model with pretrained weights from brfore on new data,
# we should trun the warm_start parameter true in model training