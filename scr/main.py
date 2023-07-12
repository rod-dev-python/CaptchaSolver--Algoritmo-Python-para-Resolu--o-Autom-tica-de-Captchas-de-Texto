from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from tratar_captcha import tratar_imagens
def quebra_nozes():
    # Importar o modelo treinado e o tradutor
    with open("rotulos_do_modelo.dat", "rb") as arquivo_tradutor:
        lb=pickle.load(arquivo_tradutor)
    modelo= load_model("modelo_treinado.hdf5")
    # Usar o modelo para resolver o captcha
    tratar_imagens("resolver", pasta_destino="resolver")
    # Ler todos os arquivos da pasta resolver
    ###################################################################################################################
    arquivos = list(paths.list_images("resolver"))
    for arquivo in arquivos:
        imagem = cv2.imread(arquivo)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        _, nova_imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY_INV)
        contornos, _ = cv2.findContours(nova_imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regiao_letras = []
        for contorno in contornos:
            (x, y, l, a) = cv2.boundingRect(contorno)
            area = cv2.contourArea(contorno)
            if area > 115:
                regiao_letras.append((x, y, l, a))
        # Ordena os intes conforme está na imagem
        regiao_letras=sorted(regiao_letras, key=lambda lista: lista[0])
        # Desenha os contornos e separar letras induviduais
        imagem_final = cv2.merge([imagem] * 3)
        previsao=[]
        i = 0
        for retangulo in regiao_letras:
            x, y, l, a = retangulo
            imagem_letra = imagem[y - 3:y + a + 3, x - 3:x + l + 3]
            # Dar a letra para a inteligênica artificial
            imagem_letra = resize_to_fit(imagem_letra, 20, 20)
            # Adicionar 4 dimensões na imagem
            imagem_letra=np.expand_dims(imagem_letra, axis=2)
            imagem_letra = np.expand_dims(imagem_letra, axis=0)
            # Prevendo a letra
            letra_prevista=modelo.predict(imagem_letra)
            letra_prevista=lb.inverse_transform(letra_prevista)[0]
            # Adicionando a letra na lista de previsão
            previsao.append(letra_prevista)
        texto_previsao="".join(previsao)
        ##############################################################################################################
        #print(texto_previsao)
        return texto_previsao
if __name__ == "__main__":
    quebra_nozes()