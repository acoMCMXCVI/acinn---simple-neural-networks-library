# acinn - simple neural network python library

AcInn je jednostavna  _high-level Python_ biblioteka napravljena za kreiranje jednostavnih sekventnih (sloj po sloj) neuronskih mreža. 
Biblioteka je napravljena na osnovu predavanja __*Anderw Nga*__, te dodatne literature 
a sve sa ciljem dubljeg razumevanja kako funkcionišu neuronske mreže. Cilj je bio napisati biblioteku koristeći samo _Python_ biblioteku _numpy_.

## Onsovni elementi biblioteke:

__Bitno je napomenuti da je biblioteka u izradi, te da je trenutno implementiran tek deo elemenata koji su 
potrebni za potpuno kreiranje i korišćenje neuronskih mreža.  
Takođe, bliblioteka prvenstveno namenjena učenju, 
i razumevanju kako funkcionišu elementi neuronskih mreža, a ne korišećenju u konkretnim projektima, i ako je to uz nju moguće.  
Elementi koji su u planu za dalju implementaciju su izlistani u nekom od narednih poglavlja, 
a u ovom poglavlju će biti izlistani elementi koji su trenutno implementirani.__

### Modeli

Sintaksa biblioteke je pisana po uzoru na sintaksu koja se koristi prilikom korišćenja ___Keras Sequential model-a___. 
(Trenutno) jedina vrsta modela koja je implementirana je simbolično nazvan __AcInn()__ model. 

Model sadrži određene metode:

#### add
```Acinn.add(layer)``` 

  Pridodaje _layer_ modelu.

__Argumenti:__ 
* __layer:__ Predstavlja _layer_ koji pridodajemo modelu

#### compile 
```Acinn.compile(initializer = 'random', loss = 'mean_squared_error', optimizer = Optimizer())``` 

Sastavlja model, odnosno vrši inicijalizaciju parametara, definiše _loss_ funckiju modela, i optimizator.

__Argumenti:__ 
* __initializer:__ Definiše kako će se inicijalizovati parametri modela
* __loss:__ Definiše koja će se funkcija gubitka koristi za model 
* __optimizer:__ Definiše koji će se optimizator koristiti za optimizaciju funkcije gubitka modela

#### fit 
```Acinn.fit(X, Y, batch_size = 32, epochs = 1, validation_split = 0., info=True)``` 

Trenira model da se prilagodi trening skupu (X, Y).

__Argumenti:__ 
* __X:__ Ulazni skup
* __Y:__ _Ground Truth_ skup
* __batch_size:__ Broj primera koji čini jedan _mini batch_
* __epochs:__ Broj epoha prilikom treninga
* __validation_split:__ Definiše procenat (_float_ od 0 do 1) trening skupa koji će se koristiti kao validacioni skup
* __info:__ Definiše da li će se informacije (gubitak i tačnost) tokom treninga prikazivati

#### evaluate 
```Acinn.evaluate(X, Y)``` 

Vrši evaluaciju modela nad dobijenim skupom (X, Y).

__Argumenti:__ 
* __X:__ Ulazni skup
* __Y:__ _Ground Truth_ skup

#### predict 
```Acinn.predict(X, in_model = False)``` 

Proračunava krajnje vrednosti sa najvećom verovatnoćom za dati skup X.

__Argumenti:__ 
* __X:__ Ulazni skup
* __in_model:__ Oređuje da li je prehodno potrebno izračunati verovatniće

#### accuracy 
```Acinn.accuracy(Y, predictions)``` 

Računa tačnost modela na osnovu _gound truth_ i predviđenih vrednosti.

__Argumenti:__ 
* __Y:__ _Ground Truth_ skup
* __predictions:__ Skup vrednosti dobijenih predviđanjem modela

#### save_weights 
```Acinn.save_weights(path)``` 

Čuva vrednosti parametara modela na disk.

__Argumenti:__ 
* __path:__ Putanja na disku gde želimo sačuvati vrednosti parametara

#### load_weights 
```Acinn.load_weights(path)``` 

Uzima vrednosti parametara modela sa diska.

__Argumenti:__ 
* __path:__ Putanja na disku odakle želimo uzeti vrednosti parametara

___

### _Layers_

_Layers_ ili slojevi su elementi čijim kombinovanjem gradimo arhitekturu modela.
I ako neuronske mreže danas poznaju veliki broj različitih slojeva, trenutno je u biblioteci implementiran samo gusto povezani _Dense (fully connected)_ sloj.

#### Dense
```layers.Dense(units, activation = 'relu', input_shape)``` 

Klasičan gusto povezani sloj.

__Argumenti:__ 
* __units:__ Definiše broj jedinica unutar sloja
* __activation:__ Definiše aktivacionu funckiju  sloja
* __input_shape:__ Definiše _shape_ ulaza za taj sloj u slučaju da je prvi sloj modela

___

### Funkcija gubitaka

Funkcija gubitka (_Loss_) je jedan od dva glavna elementa za sastavljanje (_compile_) modela. 

```Acinn.compile(loss, optimizer)``` 

Broj funkcija koje se koriste u neuronskim mrežama je velik, te je (za sada) implementirano samo par najvažnijih.

Argumenti funkcije gubitka su_
* __Y_pred:__ Skup vrednosti verovatnoća dobijenih modelom
* __Y:__ _Ground Truth_ skup

#### _Mean Squared Error_
```losses.mean_squared_error(Y_pred, Y)``` 

_Mean Squared Error_ ili srednje kvadratna funckija greške je funkcija koja se pretežno koristi kod problema regresije. 

#### _Binary Crossentropy_
```losses.binary_crossentropy(Y_pred, Y)``` 

_Binary Crossentropy_ je funkcija koja se koristi kod problema logističke regresije, odnosno klasifikacije sa dve klase. 


#### _Categorical Crossentropy_
```losses.categorical_crossentropy(Y_pred, Y)``` 

_Categorical Crossentropy_ je funkcija koja se koristi kod problema klasifikacije.

___

### _Initializers_

_Initializers_ definišu kako će se parametri inicijalizovati prilikom sastavljanja (_compile_) modela.

Ponuđeni su:

* ___Zero initializers___
* ___Relu initializers___
* ___Xavier initializers___

___

### Omptimizazor

Optimizator je jedan od dva glavna elementa za sastavljanje (_compile_) modela.

```Acinn.compile(loss, optimizer)``` 


Optimizator se definiše kao objekat klase 

``` optimizers.Optimizer(optimizer, learning_rate = 0.001, beta = 0.9, beta1 = 0.9, beta2 = 0.99, decay = 0)``` 

__Argumenti:__ 
* __optimizer:__ Definiše optimizator koji se koristi za optimizaciju funkcije gubitka modela. Mogući optimizatori su:
  * ___Stochastic Gradient Descent___
  * ___Momentum___
  * ___RMSprop___
  * ___Adam___
* __learning_rate:__ Definiše koliko stopu učenja modela
* __beta:__ Definiše parametar koji određuje ponašanje _momentum/RMS prop_ optimizatora
* __beta1 i beta2:__ Definiše parametre koji određuju ponašanje _Adam_ optimizatora
* __decay:__ Definiše opadanje stope učenja, koristeći standardnu metodu


___

## Primer korišćenja biblioteke

U ovom primeru će biti prikazan primer kreiranja modela koji rešava problem klasifikacije cveta Irisa. 
Skup podataka za treniranje u ovom primeru je skup koji se koristi kao ulazni primer za sve koji počinju da uče mašinsko učenje.
Skup sadrži po 50 primera tri različite vrste cveta irisa _(Iris setosa, Iris virginica i Iris versicolor)_, i po četiri njihove osobine.

Za ovaj primer kreirali smo model koji ima četiri skrivena sloja, dok su slojevi sačinjeni od 10, 20, 10 i 3 jedinice.
Kao optimizator je korišten _Adam_ sa vrednostima za beta1=0.9 i beta2=0.99,
a za funciju gubitka je korištena _categorical_crossentropy_ funkcija koja se koristi za problem klasifikacije.  
Model je treniran kroz 1500 epoha, sa stopom učenja learning_rate=0.005, i opadanjem decay = 0.01.

```
from acinn.models import Acinn
from acinn.optimizers import Optimizer
from acinn.layers import Dense

from utilities import load_iris_set


train_x, train_y = load_iris_set()

model = Acinn()

model.add(Dense(10, activation = 'relu', input_shape = train_x.shape[0]))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3, 'softmax'))

model.compile(initializer = 'xavier', loss = 'categorical_crossentropy', optimizer = Optimizer(optimizer = 'Adam', learning_rate=0.005, decay = 0.01))
history = model.fit(train_x, train_y, batch_size = 32, epochs = 1500, validation_split = 0.2)

model.save_weights('model')

cost_acc_train = model.evaluate(train_x, train_y)
print('model cost and acc is for train:' + str(cost_acc_train))
```

Izlaz koji dobijamo nakon pokretanja koda je:
```
Epoch 2000 / 2000 	 train loss - 0.132420 	 dev loss - 0.063776 	 dev acc - 98.321231

model cost and acc is for train:(0.11850195818248044, 97.66666666666667)
```

Dobijeni rezultati nam govore da uz pomoć ovog modela dobijamo tačnost modela na setu za trening od celih 98%.


## Elementi biblioteke koji su u pripremi

U prethodnim poglavljima je spomenuto da je samo nekolicina elemenata imnplementirana u biblioteci trenutno. 
Biblioteka je u razvoju te su mnogi elementi trenutno u pripremi. 
Neki od elemenata koji će biti implementirani su:

* Slojevi (_layers_)
  * konvolucioni slojevi 
  * _pool_ slojevi
  * _dropout_
  * _recurent_ slojevi (_RNN, GRU, LSTM_)

* Regularizatori
  * L1
  * L2
  
 * Vizuelizacija modela
 * Čuvanje na disk celokupnog modela

## Zašto ime AcInn?

Ne znam ni ja, iskreno. Zapravo, prilikom osmišljavanja naziva biblioteke, pre nego što je započet proces izrade, ova reč je imala smisla. Imala je značenje. Par nedelja kasnije, značenje je nestalo (zaboravilo se). Neke pretpostavke su da je to možda trebalo da znači __Acos' Custom Implementation of Neural Network__, ali ne zvuči toliko _WoW_ koliko je zvučalo pre, u mojoj glavi. 

Ali za sada bude to, dok se pravo značenje ne vrati :) 


