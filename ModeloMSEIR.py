from scipy.integrate import solve_ivp
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr 

mpl.style.use('seaborn')
sns.set_style("white")

####################### Modelo MSEIR inicial #################################

#Definimos los parametros iniciales del modelo 
N = 100_000
I0 = 1
S0 = N - I0
R0 = 0
E0 = 0
M0 = 0
Y0 = (M0,S0,E0,I0,R0)

CE = 4
D = 14
delta = 1/5.2
mu = 0.03


def mseir(t, y, CE, D, N, delta, mu):
    m,s,e,i,r = y
    beta = CE/N
    lambdaa = beta*i
    
    ER = s*lambdaa  #Tasa de conversion de suceptible a latentes 
    IR = delta*e    #Tasa de conversion de latentes a infectados 
    RR = i/D        #Tasa de recuperacion
    MR = mu*i       #Tasa de mortalidad 
    
    ds = -ER
    de = ER - IR
    di = IR -RR - MR
    dr = RR
    dm = MR
    
    return(dm, ds, de, di, dr)

#Resolvemos el modelo para ts dias 
ts = 60
sol = solve_ivp(mseir, [0,ts], Y0, args = (CE, D, N, delta, mu), t_eval = np.arange(0, ts, 0.125))

#Graficamos las respuestas
plt.plot(sol.t, sol.y.T, label = ['M' 'S' 'E' 'I' 'R'])
plt.legend(['M', 'S', 'E', 'I', 'R'], loc = 'upper right')
plt.show()

####################### Analisis de Sensibilidad #################################

#Definimos la version del Latin Hypercube Sampling para soportar mas de 2 dimensiones
def latin_hypercube_uniform(ranges, samples):

    parameter_names = ranges.keys()
    minimos = np.array([value[0] for key,value in ranges.items()])
    maximos = np.array([value[1] for key,value in ranges.items()])

    lower_limits, step = np.linspace(minimos, maximos, samples, endpoint=False, retstep=True)
    upper_limits = lower_limits + step

    points = np.random.default_rng().uniform(low=lower_limits, high=upper_limits).T

    for i in np.arange(1,len(parameter_names)):
        np.random.shuffle(points[i])

    return (points, parameter_names)

#Definimos el metodo para ejecutar el numero de simulaciones deseadas
def run_sims(parameters, ts):
    corridas = [] 
    for params in parameters.T:
        CE, D, I0, delta, mu = params
        N = 100_000
        S0, R0, E0, M0 = N - I0, 0, 0, 0
        Y0 = (M0, S0, E0, I0, R0)
        sol = solve_ivp(mseir,[0,ts],Y0, args=(CE,D,N,delta,mu), t_eval=np.arange(0,ts,0.125))
        corridas.append(sol.y) 

    corridas = np.dstack(corridas)

    # El shape es 5 (M,S,E,I,R) x ts x corridas
    return corridas

#Corremos las simulaciones 

#Definimos los posibles rangos para los valores de los parametros del modelo 
rangos = {'CE':(1,15), 'D':(5,25), 'I0':(1,25), 'delta':(0,1), 'mu':(0,0.5)}

n_muestras = 500 #numero de simulaciones
parametros,_ = latin_hypercube_uniform(rangos, n_muestras)
corridas = run_sims(parametros, ts) 
corridas.shape

#Grafica la simulaciones para los infectados
plt.plot(np.arange(0,ts,0.125), corridas[3,:,:], c='r', alpha=0.05)
plt.ylim([0,100_000])
plt.xlim([0,ts])
plt.show()

#Grafica por percentiles     
y = corridas[3,:,:]
n=1
perc1 = np.percentile(y, np.linspace(1, 50, num=n, endpoint=False), axis=1)
perc2 = np.percentile(y, np.linspace(50, 99, num=n+1)[1:], axis=1)
 
for p1, p2 in zip(perc1, perc2):
    plt.fill_between(np.arange(0,ts,0.125), p1,p2, alpha=1, color='lightgrey', edgecolor=None, label="99%")
 
perc1 = np.percentile(y, np.linspace(5, 50, num=n, endpoint=False), axis=1)
perc2 = np.percentile(y, np.linspace(50, 95, num=n+1)[1:], axis=1)
 
for p1, p2 in zip(perc1, perc2):
    plt.fill_between(np.arange(0,ts,0.125), p1,p2, alpha=1, color='silver', edgecolor=None, label="90%")
 
perc1 = np.percentile(y, np.linspace(25, 50, num=n, endpoint=False), axis=1)
perc2 = np.percentile(y, np.linspace(50, 75, num=n+1)[1:], axis=1)
 
for p1, p2 in zip(perc1, perc2):
    plt.fill_between(np.arange(0,ts,0.125), p1,p2, alpha=0.5, color='dimgrey', edgecolor=None, label="50%")
 
plt.plot(np.arange(0,ts,0.125), np.mean(y, axis=1), color='k', linestyle=":", label='mean')
plt.plot(np.arange(0,ts,0.125), np.median(y, axis=1), color='k', label="median")
plt.legend()
plt.ylim([0,100_000])
plt.xlim([0,ts])
plt.show()
    
####################### Statistical Screening #################################

#Calculamos las correlaciones: 

infecciosos = corridas[3,:,:] # Shape t x simulaciones
CEs = parametros[0] # shape simulaciones
Ds = parametros[1]
I0s = parametros[2]
deltas = parametros[3]
mus = parametros[4]

corr_CE = np.apply_along_axis(pearsonr,1,infecciosos, CEs)[:,0] # Descartamos p-value
corr_D =  np.apply_along_axis(pearsonr,1,infecciosos, Ds)[:,0]
corr_I0 =  np.apply_along_axis(pearsonr,1,infecciosos, I0s)[:,0]
corr_delta =  np.apply_along_axis(pearsonr,1,infecciosos, deltas)[:,0]
corr_mu =  np.apply_along_axis(pearsonr,1,infecciosos, mus)[:,0]


# Calculamos también el valor promedio de los infecciosos 
infecciosos_promedio = np.mean(infecciosos, axis=1)
simulation_time = np.arange(0,ts,0.125)

# Grafica de las series del valor promedio y las correlaciones
fig, axes = plt.subplots(2,1, sharex=True)


#Graficamos las correlaciones en el tiempo para ts
axes[0].set_xlim(0,30)
axes[0].plot(simulation_time, infecciosos_promedio, 'k')
axes[1].plot(simulation_time, corr_CE, 'g', label='CE')
axes[1].plot(simulation_time, corr_D, 'b', label='D')
axes[1].plot(simulation_time, corr_I0, 'r', label='I0')
axes[1].plot(simulation_time, corr_delta, 'c', label='delta')
axes[1].plot(simulation_time, corr_mu, 'm', label='mu')
axes[1].legend()
axes[1].set_xlabel('Días')

fig.tight_layout()
plt.show()

#tiempo = np.argwhere(simulation_time == 15)
#print(tiempo)
#los primeros 20 dias se encuentran en los primeros 120 elementos


#Graficamos los boxplot de las correlaciones para los primeros 20 dias 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot([corr_CE[:120], corr_D[:120], corr_I0[:120], corr_delta[:120], corr_mu[:120]])
ax.set_xticklabels(['CE', 'D', 'I0', 'delta', 'mu'])
fig.tight_layout()
plt.show()


