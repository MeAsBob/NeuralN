from PIL import Image , ImageTk
import numpy as np
import tkinter as tk
from tkinter import ttk , filedialog , Label , Canvas
import math
import random
import os


def generate_one_hot(letter, choices):
    one_hot = np.zeros(len(choices), dtype=int)
    index = choices.index(letter)  # Znajdź indeks dla danej litery
    one_hot[index] = 1  # Ustaw odpowiednią pozycję na 1
    return one_hot


def open_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Wybierz plik", filetypes=(("Obrazy JPG", "*.jpg"),("Obrazy PNG", "*.png"), ("Wszystkie pliki", "*.*")))
    return file_path


def imagine_resize(photo):
    image = Image.open(photo)
    image_resized = image.resize((32, 32))
    return image_resized


def convolve2d(image, kernel, stride=1, padding=0):
    image_padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    # Wymiary obrazu i jądra
    i_h, i_w, i_c = image_padded.shape
    k_h, k_w, k_c = kernel.shape  # Wymiary wyjściowej mapy aktywacji
    output_height = (i_h - k_h) // stride + 1
    output_width = (i_w - k_w) // stride + 1
    output = np.zeros((output_height, output_width, i_c))  # Przesuń filtr po obrazie
    for y in range(0, output_height):
        for x in range(0, output_width):
            for c in range(i_c):  # Przetwarzamy każdy kanał
                output[y, x, c] = np.sum(
                    image_padded[y * stride:y * stride + k_h, x * stride:x * stride + k_w, c] * kernel[:, :, c])

    return output


def activaction_relu(data: object) -> object:
    data = np.maximum(0, data)
    data = np.round(data, 2)
    return data

def open_okno_2():
    okno_1.destroy()
    okno_2 = tk.Tk()
    okno_2.configure(bg="grey")
    okno_2.geometry("400x500")
    okno_2.title("Uczenie Sieci Neuronowej")
    entry = tk.Entry(okno_2, width=30)  # Tworzy pole tekstowe
    entry.pack(pady=10)
    start = tk.Button(
        okno_2,
        text="Start",
        command=lambda: calosc_Uczenia(entry.get(), okno_2, 0)
    )
    start.pack(pady=15)
    end = tk.Button(
        okno_2,
        text="Wróć",
        command=lambda: powrot(okno_2)
    )
    end.pack(pady=15)


def open_okno_3():
    okno_1.destroy()
    okno_3 = tk.Tk()
    okno_3.configure(bg="grey")
    okno_3.geometry("300x450")
    okno_3.title("Sprawdź znak")
    start = tk.Button(
        okno_3,
        text="Start",
        command=lambda: calosc_Uczenia(1, okno_3, 1)
    )
    start.pack(pady=15)
    end = tk.Button(
        okno_3,
        text="Wróć",
        command=lambda: powrot(okno_3)
    )
    end.pack(pady=15)

def pokaz_obraz(konkretnyplik,can):
    # Wyrzucanie obrazka na canvas
    can.delete("all")
    can.pack()
    obraz = Image.open(konkretnyplik).resize((200, 200))
    photo = ImageTk.PhotoImage(obraz)
    can.create_image(0, 0, anchor="nw", image=photo)
    can.image = photo


def powrot(gdzie):
    gdzie.destroy()  # Zamyka bieżące okno
    if gdzie != okno_1:  # Jeśli wracamy z innego okna niż okno_1
        otworz_okno_1()  # Otwórz okno_1 ponownie
    if gdzie == okno_1:  # Jeśli wracamy z innego okna niż okno_1
        exit()


def otworz_okno_1():
    global okno_1
    okno_1 = tk.Tk()
    okno_1.configure(bg="grey")
    okno_1.title("Wybierz operacje")
    okno_1.geometry("300x300")
    button_okno_2 = tk.Button(okno_1, text="Uczenie Sieci", command=open_okno_2)
    button_okno_2.pack(pady=20)
    button_okno_3 = tk.Button(okno_1, text="Sprawdź znak", command=open_okno_3)
    button_okno_3.pack(pady=20)
    button_okno_4 = tk.Button(okno_1, text="Zamknij Program", command=lambda: powrot(okno_1))
    button_okno_4.pack(pady=20)

    okno_1.mainloop()



def calosc_Uczenia(Ile, gdzie, tryb):
    ilosc_neur = 1024
    canvas = Canvas(gdzie, width=200, height=200, bg="white")
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
    else:
        waga_0 = np.random.randn(ilosc_neur, 3072) * np.sqrt(2 / 3072)
        waga_0 = np.round(waga_0, 2)
        waga_1 = np.random.randn(ilosc_neur, ilosc_neur) * np.sqrt(2 / 8)
        waga_1 = np.round(waga_1, 2)
        waga_2 = np.random.randn(4, ilosc_neur) * np.sqrt(2 / 8)
        waga_2 = np.round(waga_2, 2)
        bias_0 = np.zeros(ilosc_neur)  # Bias dla warstwy pierwszej
        bias_1 = np.zeros(ilosc_neur)  # Bias dla warstwy drugiej
        bias_2 = np.zeros(4)  # Bias dla warstwy wyjściowej
        print("Utworzono nowe dane")

    epochs = int(Ile)  # Liczba epok
    if (tryb == 0):
        learning_rate = 0.0001
        progress = ttk.Progressbar(gdzie, orient="horizontal", length=epochs, mode="determinate")
        progress.pack(pady=20)
        choices_letter = ['A', 'B', 'C', 'D']  # Pierwszy przedział
        choices_number = {'A': [1, 2, 7, 17,21, 30],  # Drugi przedział zależny od pierwszego
                          'B': [1, 2, 20, 21, 22, 23, 33, 36, 41],
                          'C': [2, 4, 12],
                          'D': [1, 6]}
        total_loss = 0
        dobrze = 0
        calosc = 0
    else:
        learning_rate = 0

    for epoch in range(epochs):
        if(tryb == 0):
            letter = random.choice(choices_letter)
            number = random.choice(choices_number[letter])
            one_hot_letter = generate_one_hot(letter, choices_letter)

            scieszka = fr"classification\{letter}{number}"
            # Pobranie listy plików .jpg w folderze
            files = os.listdir(scieszka)
            jpg_files = [f for f in files if f.endswith('.jpg')]

            # Losowy plik
            liczba = random.choice(jpg_files)  # Losowy plik .jpg
            konkretnyplik = os.path.join(scieszka, liczba)
        else:
            konkretnyplik = open_file()
        print(konkretnyplik)
        kernel = np.ones((1, 1, 3))
        image_ref = imagine_resize(konkretnyplik)
        image_ref = convolve2d(image_ref, kernel)
        image_ref = image_ref.flatten()
        image_data = image_ref.reshape((32, 32, 3)).astype(np.uint8)

        pokaz_obraz(konkretnyplik, canvas)


        E = math.e
        hidden = np.zeros((ilosc_neur, 2))
        end = np.zeros(4)
        active_0 = np.zeros(ilosc_neur)
        active_1 = np.zeros(ilosc_neur)

        i: int = 0
        for y in range(image_data.shape[0]):  # Przechodzimy przez wysokość
            for x in range(image_data.shape[1]):  # Przechodzimy przez szerokość

                X[i] = 1 - (image_data[y, x, 0] / 255)  # Kanał czerwony
                X[i] = np.round(X[i], 2)

                X[i + 1024] = 1 - (image_data[y, x, 1] / 255)  # Kanał zielony
                X[i + 1024] = np.round(X[i + 1024], 2)

                X[i + 2048] = 1 - (image_data[y, x, 2] / 255)  # Kanał niebieski
                X[i + 2048] = np.round(X[i + 2048], 2)

                i = i + 1

        a: int = 0
        for a in range(ilosc_neur):
            hidden[a][0] = np.dot(waga_0[a], X) + bias_0[a]  # Obliczanie wartości dla hidden
            active_0[a] = activaction_relu(hidden[a][0])  # Przekształcanie przez funkcję aktywacji

        b: int = 0
        for b in range(ilosc_neur):
            hidden[b][1] = np.dot(waga_1[b], active_0) + bias_1[b]
            active_1[b] = activaction_relu(hidden[b][1])  # Przekształcanie przez funkcję aktywacji


        for c in range(4):
            end[c] = np.dot(waga_2[c], active_1) + bias_2[c]
        end = end - np.max(end)  # Stabilizacja numeryczna
        end_value = np.exp(end) / np.sum(np.exp(end))

        print(end_value)
        if (tryb == 0):
            strata = -np.sum(one_hot_letter * np.log(end_value + 1e-15))  # Tutaj poprawiłem nwm czy dobrze
            print(strata)  # Im większa tym gorzej
            total_loss += strata

            # Obliczanie gradientów wstecznej propagacji
            gradien = end_value - one_hot_letter  # Gradient błędu warstwy wyjściowej

            # Gradient dla warstwy wyjściowej
            d_hidden1 = np.dot(waga_2.T, gradien)  # Rozmiar (8,)
            d_active1 = np.where(active_1 > 0, 1, 0)  # Rozmiar (8,) – pochodna ReLU
            d_hidden1 *= d_active1

            # Gradient dla pierwszej warstwy ukrytej
            d_active0 = np.where(active_0 > 0, 1, 0)  # Rozmiar (8,) – pochodna ReLU
            d_hidden0 = np.dot(waga_1.T, d_hidden1)
            d_hidden0 *= d_active0  # Element-wise mnożenie

            # Gradienty biasów
            d_bias_2 = gradien
            d_bias_1 = d_hidden1
            d_bias_0 = d_hidden0

            # Obliczanie delty wag
            d_waga_2 = np.outer(gradien, active_1)
            d_waga_1 = np.outer(d_hidden1, active_0)
            d_waga_0 = np.outer(d_hidden0, X)

            # Aktualizacja wag
            waga_2 -= learning_rate * d_waga_2
            waga_1 -= learning_rate * d_waga_1
            waga_0 -= learning_rate * d_waga_0

            # Aktualizacja biasów
            bias_2 -= learning_rate * d_bias_2
            bias_1 -= learning_rate * d_bias_1
            bias_0 -= learning_rate * d_bias_0

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X):.4f}")
            # Sprawdzanie poprwncych
            actual_class = np.argmax(one_hot_letter)




        predicted_class = np.argmax(end_value)
        if(tryb == 1):
            if(predicted_class == 0):
                label_1 = tk.Label(gdzie, text="A – znaki drogowe ostrzegawcze")
            elif (predicted_class == 1):
                label_1 = tk.Label(gdzie, text="B – znaki drogowe zakazu")
            elif(predicted_class == 2):
                label_1 = tk.Label(gdzie, text="C – znaki drogowe nakazu")
            elif(predicted_class == 3):
                label_1 = tk.Label(gdzie, text="D – znaki drogowe informacyjne")
            label_1.pack(pady=20)


        if tryb == 0:
            if predicted_class == actual_class:
                dobrze += 1
                print("Dobrze")
            else:
                print("źle")
            calosc += 1
            if epoch == epochs / 2:
                learning_rate = learning_rate / 10
                print("learning rate został zmniejszony")

            progress["value"] = epoch  # Ustawianie wartości paska
            gdzie.update_idletasks()
            print(f"{dobrze}/{calosc}")
            print("--------------------------------")


    if(tryb == 0):
        np.savez("waga_bias.npz", matrix0=waga_0, matrix1=waga_1, matrix2=waga_2, matrix3=bias_0, matrix4=bias_1, matrix5=bias_2)
        print(dobrze / calosc)
        print("Dane powinny zostać zapisane w pliku")
        label_1 = tk.Label(gdzie, text="Poprawność testowania = "+f"{dobrze}/{calosc}")
        label_1.pack(pady=20)








otworz_okno_1()


