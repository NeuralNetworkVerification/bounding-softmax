import pickle
import numpy as np
from keras.datasets import mnist
from scipy.special import softmax


NUM_INPUT = 100
NUM_ENSEMBLE = 5
EPSILON = [0.008, 0.012, 0.016]
# EPSILON = [0.016]
NUM_LABEL = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255

print('Computing NLL and Brier scores...')
for eps in EPSILON:
    NLLs = []
    Briers = []
    for n in range(NUM_INPUT):
        oracle = y_test[n]
        lbs = []
        ubs = []
        for m in range(NUM_ENSEMBLE):
            # with open(f"./bounds/bounds_net{m}_ind{n}_eps{eps}.pickle", 'rb') as fp:
            # with open(f"./bounds/bounds_netlarge-{m}_ind{n}_eps{eps}.pickle", 'rb') as fp:
            with open(f"./bounds/bounds_netrobust-large-{m}_ind{n}_eps{eps}.pickle", 'rb') as fp:
                bounds = pickle.load(fp)
            lbs.append(bounds["lbs"][-1])
            ubs.append(bounds["ubs"][-1])
        lbs = np.asarray(lbs)
        ubs = np.asarray(ubs)

        # compute NLL
        lb = lbs[:, oracle]
        NLL = -np.log(lb.mean())
        NLLs.append(NLL)

        # compute Brier
        # lb = lbs[:, oracle]
        py = lb.mean()
        temps = []
        for label in range(NUM_LABEL):
            pk_ = lbs[:, label].mean()
            _pk = ubs[:, label].mean()
            if label == oracle:
                pk = py
            else:
                pk = _pk
            temp = (pk_ + _pk) * pk - pk_ * _pk
            temps.append(temp)
        temps = np.asarray(temps)
        Brier = -2 * py + np.sum(temps) + 1
        Briers.append(Brier)

    print('epsilon:', eps)
    # NLL score over all inputs
    NLLs = np.asarray(NLLs)
    NLL = NLLs.mean()
    # NLL = np.ma.masked_invalid(NLLs).mean()
    print('NLL:', NLL)

    # Brier score over all inputs
    Briers = np.asarray(Briers)
    Brier = Briers.mean()
    print('Brier:', Brier)
# print('NLL and Brier scores computed.')


def inference(model, image):
    w1, b1 = model["w1"], model["b1"]
    w2, b2 = model["w2"], model["b2"]
    w3, b3 = model["w3"], model["b3"]
    w4, b4 = model["w4"], model["b4"]
    tmp = np.dot(image, w1) + b1
    tmp = np.clip(tmp, a_min=0, a_max=None)
    tmp = np.dot(tmp, w2) + b2
    tmp = np.clip(tmp, a_min=0, a_max=None)
    tmp = np.dot(tmp, w3) + b3
    tmp = np.clip(tmp, a_min=0, a_max=None)
    logits = np.dot(tmp, w4) + b4
    # logits = np.dot(tmp, w3) + b3
    probabilities = softmax(logits)
    return probabilities


print('\nComputing clean NLL and Brier scores...')
clean_NLLs = []
clean_Briers = []
for n in range(NUM_INPUT):
    oracle = y_test[n]
    image = x_test[n]
    probs = []
    for m in range(NUM_ENSEMBLE):
        # model = np.load('./networks/mnist-%d.npz' % m)
        # model = np.load('./networks/mnist-large-%d.npz' % m)
        model = np.load('./networks/robust_mnist-large-%d.npz' % m)
        prob = inference(model, image)
        probs.append(prob)
    probs = np.asarray(probs)

    # compute clean NLL
    py = probs[:, oracle].mean()
    clean_NLL = -np.log(py)
    clean_NLLs.append(clean_NLL)

    # compute clean Brier
    pk = [probs[:, label].mean() for label in range(NUM_LABEL) if label != oracle]
    clean_Brier = np.square(1 - py) + np.sum(np.square(pk))
    clean_Briers.append(clean_Brier)

# clean NLL score over all inputs
clean_NLLs = np.asarray(clean_NLLs)
clean_NLL = clean_NLLs.mean()
print('clean NLL:', clean_NLL)

# Brier score over all inputs
clean_Briers = np.asarray(clean_Briers)
clean_Brier = clean_Briers.mean()
print('clean Brier:', clean_Brier)

# clean_NLL_Brier = dict()
# clean_NLL_Brier['NLL'] = clean_NLLs
# clean_NLL_Brier['Brier'] = clean_Briers
# with open('clean_NLL_Brier_mnist100.pkl', 'wb') as f:
#     pickle.dump(clean_NLL_Brier, f)

# print('Clean NLL and Brier scores computed.')

