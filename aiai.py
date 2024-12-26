from PIL import Image
import numpy as np
import math
import random
import os

def generate_one_hot(letter, choices):
    one_hot = np.zeros(len(choices), dtype=int)
    index = choices.index(letter)  # Znajdź indeks dla danej litery
    one_hot[index] = 1             # Ustaw odpowiednią pozycję na 1
    return one_hot

def imagine_resize(photo):
    image = Image.open(photo)
    image_resized = image.resize((32, 32))
    return image_resized

def convolve2d(image, kernel, stride=1, padding=0):
    image_padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    # Wymiary obrazu i jądra
    i_h, i_w, i_c = image_padded.shape
    k_h, k_w, k_c = kernel.shape        # Wymiary wyjściowej mapy aktywacji
    output_height = (i_h - k_h) // stride + 1
    output_width = (i_w - k_w) // stride + 1
    output = np.zeros((output_height, output_width, i_c))    # Przesuń filtr po obrazie
    for y in range(0, output_height):
        for x in range(0, output_width):
            for c in range(i_c):  # Przetwarzamy każdy kanał
                output[y, x, c] = np.sum(image_padded[y*stride:y*stride+k_h, x*stride:x*stride+k_w, c]* kernel[:, :, c])

    return output

def activaction_relu(data: object) -> object:
    data = np.maximum(0,data)
    data = np.round(data, 2)
    return data


X = np.zeros(3072, dtype=np.float32)
file = "waga_bias.npz"
if os.path.exists(file):
    zapisane_macierze = np.load(file)
    waga_0 = zapisane_macierze["matrix0"]
    waga_1 = zapisane_macierze["matrix1"]
    waga_2 = zapisane_macierze["matrix2"]
    bias_0 = zapisane_macierze["matrix3"]
    bias_1 = zapisane_macierze["matrix4"]
    bias_2 = zapisane_macierze["matrix5"]
    zapisane_macierze.close()  # Zamknięcie pliku
    print("Wczytano dane z pliku")
    print(bias_0)
else:
    waga_0 = np.random.randn(8, 3072) * np.sqrt(2 / 3072)#waga_0 = np.random.uniform(-1, 1, (8, 3072))
    waga_0 = np.round(waga_0, 2)
    waga_1 = np.random.randn(8, 8) * np.sqrt(2 / 8)#waga_1 = np.random.uniform(-1, 1, (8, 8))
    waga_1 = np.round(waga_1, 2)
    waga_2 = np.random.randn(4, 8) * np.sqrt(2 / 8)#waga_2 = np.random.uniform(-1, 1, (4, 8))
    waga_2 = np.round(waga_2, 2)
    bias_0 = np.zeros(8)  # Bias dla warstwy pierwszej
    bias_1 = np.zeros(8)  # Bias dla warstwy drugiej
    bias_2 = np.zeros(4)  # Bias dla warstwy wyjściowej
    print("Utworzono nowe dane")



learning_rate = 0.01
epochs = 20000  # Liczba epok
choices_letter = ['A', 'B', 'C', 'D']  # Pierwszy przedział
choices_number = {'A': [1, 2, 7,21,30],      # Drugi przedział zależny od pierwszego
                  'B': [1, 2, 20,21,22,23,33,36,41],
                  'C': [2, 4, 12],
                  'D': [1, 6]}
total_loss = 0
dobrze = 0
calosc = 0

for epoch in range(epochs):

    letter = random.choice(choices_letter)
    number = random.choice(choices_number[letter])
    one_hot_letter = generate_one_hot(letter, choices_letter)

    scieszka = fr"C:\Users\adrian\PycharmProjects\NeuralN\classification\{letter}{number}"
    # Pobranie listy plików .jpg w folderze
    files = os.listdir(scieszka)
    jpg_files = [f for f in files if f.endswith('.jpg')]

    # Losowy plik
    liczba = random.choice(jpg_files)  # Losowy plik .jpg
    konkretnyplik = os.path.join(scieszka, liczba)
    print(konkretnyplik)
    kernel = np.ones((1, 1, 3))
    Image_ref=imagine_resize(konkretnyplik)
    Image_ref=convolve2d(Image_ref, kernel)
    Image_ref=Image_ref.flatten()
    image_data = Image_ref.reshape((32, 32, 3)).astype(np.uint8)

    E = math.e
    hidden = np.zeros ((8,2))
    end = np.zeros (4)
    active_0 = np.zeros (8)
    active_1 = np.zeros (8)





    i:int = 0
    for y in range(image_data.shape[0]):  # Przechodzimy przez wysokość
        for x in range(image_data.shape[1]):  # Przechodzimy przez szerokość

            X[i] = 1 - (image_data[y, x, 0] / 255)  # Kanał czerwony
            X[i] = np.round(X[i], 2)

            X[i+1024] = 1 - (image_data[y, x, 1] / 255)  # Kanał zielony
            X[i+1024] = np.round(X[i+1024], 2)

            X[i+2048] = 1 - (image_data[y, x, 2] / 255) # Kanał niebieski
            X[i+2048] = np.round(X[i+2048], 2)

            i=i+1

    a:int = 0
    for a in range(8):
        hidden[a][0] = np.dot(waga_0[a], X) + bias_0[a]  # Obliczanie wartości dla hidden
        active_0[a] = activaction_relu(hidden[a][0])  # Przekształcanie przez funkcję aktywacji

    b:int = 0
    for b in range(8):
        hidden[b][1] = np.dot(waga_1[b], active_0) + bias_1[b]
        active_1[b] = activaction_relu(hidden[b][1])  # Przekształcanie przez funkcję aktywacji

    c:int = 0
    for c in range(4):
        end[c] = np.dot(waga_2[c], active_1) + bias_2[c]
        end[c] = E**end[c]

    end_base = sum(end)
    end_value = np.zeros (4)

    d:int = 0
    for d in range(4):
        end_value[d] = (end[d] / end_base)
    print(end_value)
    strata = -(math.log(end_value[0]) * one_hot_letter[0] + math.log(end_value[1]) * one_hot_letter[1] + math.log(end_value[2]) * one_hot_letter[2] + math.log(end_value[3]) * one_hot_letter[3])
    print(strata) # Im większa tym gorzej
    total_loss += strata

    #Wsteczna propagacja , albo to co próbuje zrobić
    # Obliczanie gradientów wstecznej propagacji
    Gradien = end_value - one_hot_letter  # Gradient błędu warstwy wyjściowej


    # Gradient dla warstwy wyjściowej
    d_hidden1 = np.dot(waga_2.T, Gradien)  # Rozmiar (8,)
    d_active1 = np.where(active_1 > 0, 1, 0)  # Rozmiar (8,) – pochodna ReLU
    d_hidden1 *= d_active1  # Element-wise mnożenie

    # Gradient dla pierwszej warstwy ukrytej
    d_active0 = np.where(active_0 > 0, 1, 0)  # Rozmiar (8,) – pochodna ReLU
    d_hidden0 = np.dot(waga_1.T, d_hidden1)  # Rozmiar (8,)
    d_hidden0 *= d_active0  # Element-wise mnożenie

    # Gradienty biasów
    d_bias_2 = Gradien  # Rozmiar (4,)
    d_bias_1 = d_hidden1  # Rozmiar (8,)
    d_bias_0 = d_hidden0  # Rozmiar (8,)

    # Obliczanie delty wag
    d_waga_2 = np.outer(Gradien, active_1)  # Kształt (4, 8)
    d_waga_1 = np.outer(d_hidden1, active_0)  # Kształt (8, 8)
    d_waga_0 = np.outer(d_hidden0, X)  # Kształt (8, 3072)

    # Aktualizacja wag
    waga_2 -= learning_rate * d_waga_2
    waga_1 -= learning_rate * d_waga_1
    waga_0 -= learning_rate * d_waga_0

    # Aktualizacja biasów
    bias_2 -= learning_rate * d_bias_2
    bias_1 -= learning_rate * d_bias_1
    bias_0 -= learning_rate * d_bias_0

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X):.4f}")
    #Sprawdzanie poprwncych
    predicted_class = np.argmax(end_value)
    actual_class = np.argmax(one_hot_letter)

    # Sprawdź, czy przewidywanie jest poprawne
    if predicted_class == actual_class:
        dobrze += 1
    calosc += 1

    if epoch == epochs/2:
        learning_rate = learning_rate/10
        print("learning rate został zmniejszony")

    print(f"{dobrze}/{calosc}")
    print("--------------------------------")

np.savez("waga_bias.npz", matrix0=waga_0, matrix1=waga_1,matrix2=waga_2, matrix3=bias_0,matrix4=bias_1,matrix5=bias_2)
print(dobrze/calosc)
print("Dane powinny zostać zapisane")
