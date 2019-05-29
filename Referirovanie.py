#!/usr/bin/env python
# coding: utf-8

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import pymorphy2
def norm(x):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(x)[0]
    return p.normal_form


import re
from nltk.tokenize import sent_tokenize

def tokenize(text):
    sentences = []
    f=0
    for sents in sent_tokenize(text):
        sents.strip()
        sents = sents.replace('\n',' ')
        if f:
            s = sentences[-1]
            sentences.remove(s)
            sentences.append(s+' '+sents)
        else:
            sentences.append(sents)
        if(sents[-4:]=='т.е.'):
            f=1
        else:
            f=0
    return sentences
from nltk.corpus import stopwords
import string
stop_words = stopwords.words('russian')
def tokenize_me(file_text):
    #применили токенизацию
    tokens = nltk.word_tokenize(file_text)
 
    #удалили пунктуацию
    tokens = [i for i in tokens if ( i not in string.punctuation )]
    stop = ['что', 'это','от' ,'так', 'вот', 'быть', 'как', 'в', 'к', 'на','она','он','мы''-','—','–','нее']
    #удалили стоп-слова
    tokens = [i.lower() for i in tokens if ( i not in stop_words and i not in stop and tokens.count(i)==1)]
    for i in stop:
        if i in tokens:
            tokens.remove(i)
    #почистили сами слова
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]
 
    return tokens
def in_tokens(stem, tokens):
    for tok in tokens:
        if(tok.find(stem)>-1):
            return True
    return False
def in_tokens_2(dic, tokens):
    for d in dic:
        for tok in tokens:
            if(tok==d):
                return True
    return False



import re
import unittest
# определяем стем слова - его основу

class Stemmer:
    # Гласные и согласные.
    _vowel = "[аеиоуыэюя]"
    _non_vowel = "[^аеиоуыэюя]"

    # Word regions.
    _re_rv = re.compile(_vowel)
    _re_r1 = re.compile(_vowel + _non_vowel)

    # Окончания.
    _re_perfective_gerund = re.compile(
        r"(((?P<ignore>[ая])(в|вши|вшись))|(ив|ивши|ившись|ыв|ывши|ывшись))$"
    )
    _re_adjective = re.compile(
        r"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|"
        r"ую|юю|ая|яя|ою|ею)$"
    )
    _re_participle = re.compile(
        r"(((?P<ignore>[ая])(ем|нн|вш|ющ|щ))|(ивш|ывш|ующ))$"
    )
    _re_reflexive = re.compile(
        r"(ся|сь)$"
    )
    _re_verb = re.compile(
        r"(((?P<ignore>[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|"
        r"нно))|(ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|"
        r"ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю))$"
    )
    _re_noun = re.compile(
        r"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|"
        r"ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$"
    )
    _re_superlative = re.compile(
        r"(ейш|ейше)$"
    )
    _re_derivational = re.compile(
        r"(ост|ость)$"
    )
    _re_i = re.compile(
        r"и$"
    )
    _re_nn = re.compile(
        r"((?<=н)н)$"
    )
    _re_ = re.compile(
        r"ь$"
    )

    def stem(self, word):

        rv_pos, r2_pos = self._find_rv(word), self._find_r2(word)
        word = self._step_1(word, rv_pos)
        word = self._step_2(word, rv_pos)
        word = self._step_3(word, r2_pos)
        word = self._step_4(word, rv_pos)
        return word

    def _find_rv(self, word):
   

        rv_match = self._re_rv.search(word)
        if not rv_match:
            return len(word)
        return rv_match.end()

    def _find_r2(self, word):

        r1_match = self._re_r1.search(word)
        if not r1_match:
            return len(word)
        r2_match = self._re_r1.search(word, r1_match.end())
        if not r2_match:
            return len(word)
        return r2_match.end()

    def _cut(self, word, ending, pos):


        match = ending.search(word, pos)
        if match:
            try:
                ignore = match.group("ignore") or ""
            except IndexError:
                # No ignored characters in pattern.
                return True, word[:match.start()]
            else:
                # Do not cut ignored part.
                return True, word[:match.start() + len(ignore)]
        else:
            return False, word

    def _step_1(self, word, rv_pos):
        match, word = self._cut(word, self._re_perfective_gerund, rv_pos)
        if match:
            return word
        _, word = self._cut(word, self._re_reflexive, rv_pos)
        match, word = self._cut(word, self._re_adjective, rv_pos)
        if match:
            _, word = self._cut(word, self._re_participle, rv_pos)
            return word
        match, word = self._cut(word, self._re_verb, rv_pos)
        if match:
            return word
        _, word = self._cut(word, self._re_noun, rv_pos)
        return word

    def _step_2(self, word, rv_pos):
        _, word = self._cut(word, self._re_i, rv_pos)
        return word

    def _step_3(self, word, r2_pos):
        _, word = self._cut(word, self._re_derivational, r2_pos)
        return word

    def _step_4(self, word, rv_pos):
        _, word = self._cut(word, self._re_superlative, rv_pos)
        match, word = self._cut(word, self._re_nn, rv_pos)
        if not match:
            _, word = self._cut(word, self._re_, rv_pos)
        return word
    




import os
import sys
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora import WikiCorpus
#Загрузка уже обученной модели
m=Word2Vec.load('wiki_w2v_new.model')

def in_tokens(stem, tokens):
    for tok in tokens:
        if(tok.find(stem)>-1):
            return True
    return False
def in_tokens_2(dic, tokens):
    for d in dic:
        for tok in tokens:
            if(tok==d):
                return True
    return False


#Кнопаем для интерфейса
import sys
# Импортируем наш интерфейс
from interface22 import *
from PyQt5 import QtCore, QtGui, QtWidgets

class MyWin(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        bar = self.menuBar()
        file_menu = bar.addMenu('File')
        
        # добавляем экшн к меню
        close_action = QtWidgets.QAction('Close', self)
        file_menu.addAction(close_action)

        #используем `connect` метож
        close_action.triggered.connect(self.close)
    
        self.ui.pushButton.clicked.connect(self.DomainCheck)   
    def close_application(self):
        sys.exit()
    def on_Button_clicked(self, checked=None):
        if checked==None: return
        dialog = QDialog()
        dialog.ui = Ui_MyDialog()
        dialog.ui.setupUi(dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.exec_()
    # Пока пустая функция которая выполняется
    # при нажатии на кнопку                  
    def DomainCheck(self):
        # Очищаем второе текстовое поле
        self.ui.textEdit_2.setText("")
        # В переменную stroki получаем текст из левого поля ввода
        stroki=self.ui.textEdit.toPlainText() 
        # Получаем массив строк разделив текст по знаку переноса строки 
        sentences = tokenize(stroki)
        print(sentences)
        title=self.ui.textEdit_3.toPlainText() 
        title_words = tokenize_me(title)
        passages = []
        i=0
        n=4
        for i in range(len(sentences)-n):
            passages.append(' '.join(sentences[i:i+n]))
        semantic_dict={}
        for word in title_words:
            w = norm(word)
            semantic_dict[w] = []
            try:
                vec = m[w]
            except Exception:
                    continue
            for t in m.most_similar(positive=[w],topn=10):
                s = str(t[0])
                print(s)
                semantic_dict[w].append(s)
        r=[]
        u_w_q = len(title_words)
        for pas in passages:
            pas_tokens=[]
            dsi = 0

            if('–' in pas or '-' in pas):
                dsi = 1
            for  w in tokenize_me(pas):
                pas_tokens.append(w)
            u_k_w_q = len([i for i in title_words if (i in pas_tokens)])
            st = Stemmer()
            norms = [norm(i) for i in pas_tokens]
            t_r = len([1 for i in title_words if(in_tokens(st.stem(i),pas_tokens))])
            t_s = len([i for i in title_words if(in_tokens_2(semantic_dict[norm(i)],norms))])
            print(u_k_w_q)
            print(t_r)
            print(t_s)
            print()
            r.append(((4*u_k_w_q+1.8*(t_r - u_k_w_q)+0.5*t_s)/u_w_q +0.1*dsi)/(4+1.8+0.1+0.5))
        clasters=[]
        claster=[]
        cl_nums=[]
        cl_num=[]
        sums = []
        for i in range(len(passages)-4*n):
            s=0
            claster.append(passages[i])
            cl_num.append(i)
            s+=r[i]
            for j in range(i+n,len(passages)-3*n):
                claster.append(passages[j])
                cl_num.append(j)
                s+=r[j]
                for k in range(j+n,len(passages)-2*n):
                    claster.append(passages[k])
                    cl_num.append(k)
                    s+=r[k]
                    ind = r[k+n:].index(max(r[k+n:]))
                    s+=r[k+n+ind]
                    claster.append(passages[k+n+ind])
                    cl_num.append(k+n+ind)
                    clasters.append(claster)
                    cl_nums.append(cl_num)
                    sums.append(s)
                    claster = claster[:2]
                    cl_num = cl_num[:2]
                    s-=r[k+n+ind]
                    s-=r[k]
                claster = claster[:1]
                cl_num = cl_num[:1]
                s-=r[j]
            claster=[]
            cl_num=[]
            s=0
        if not(clasters):
            for pas in passages:
                claster.append(pas)
            clasters.append(claster)
            max_index = 0
        else:
             max_index = sums.index(max(sums))        
        print(clasters)
        m_claster = clasters[max_index]
        annotation=''
        r_p = [r[passages.index(pas)] for pas in m_claster]
        i = r_p.index(max(r_p))
        print(r_p)
        print(i)
        annotation+=m_claster[i]
        l = 500 #длина пассажа
        while True:
            if(len(annotation)>l):
                annotation = annotation[:l]
                annotation = annotation[:annotation.rfind(' ')]+'...'
                break
            elif(len(annotation)>l-100):
                break
            else:
                if(i==0):
                    annotation+=m_claster[i+1]
                elif(i==3):
                    ann=''
                    ann+=m_claster[i-1]+annotation
                    annotation=ann
                else:
                    if(r_p[i-1]>r_p[i+1]):
                        nn=''
                        ann+=m_claster[i-1]+annotation
                        annotation=ann
                    else:
                        annotation+=m_claster[i+1]
        print(annotation)   
        self.ui.textEdit_2.setText(annotation)
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    
    wgt = QtWidgets.QLabel()
    wgt.setText("   Данная программа позволяет реферировать документы. \n Чтобы получить аннотацию документа вставте текст в левое окошко, заполните название документа и нажмите кнопку Реферировать.\n Все права защищены. \n КубГУ, 2019.")
    wgt.resize(950, 150)
    wgt.setWindowTitle('О программе')
    
    bar = myapp.menuBar()
    hell_menu = bar.addMenu('About')
    showWidgetAction = QtWidgets.QAction('&Справка', myapp)
    showWidgetAction.triggered.connect(wgt.show)
    hell_menu.addAction(showWidgetAction)
    sys.exit(app.exec_())




