from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import time

seed = 0 # Durch einen festen Seedwert wird die Reproduzierbarkeit sichergestellt.

# Rückrufs-Funktion, damit die Gewinnrate berechnet wird.
def gewinnraten_evaluierung(model, umgebung, anzahl_spiele):
    siege = 0
    spielbelohnung_liste = []
    
    for _ in range(anzahl_spiele):
        obs, info = umgebung.reset(seed=seed+_)
        fertig = False
        spielbelohnung = 0
        while not fertig:
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, fertig, truncated, info = umgebung.step(action)
          spielbelohnung += reward
          
          if fertig:
              spielbelohnung_liste.append(spielbelohnung)
              if info['is_success']:
                  siege += 1

        gewinnrate = siege / anzahl_spiele
    
    return spielbelohnung_liste, gewinnrate

def umgebung_importieren(belohnungsfunktion):
    # Entsprechende Umgebung wird importiert.
    if "B2" in belohnungsfunktion:
        from Belohnungsfunktion_2 import KäsekästchenEnv
    elif "B3" in belohnungsfunktion:
        from Belohnungsfunktion_3 import KäsekästchenEnv
    else:
        from Belohnungsfunktion_1 import KäsekästchenEnv
    
    # Evaluierungsumgebung wird importiert.
    from Evaluierungsumgebung import KäsekästchenEvaluierungEnv
    
    return KäsekästchenEnv, KäsekästchenEvaluierungEnv

# Nutzer wird nach Modus gefragt
modus = input("""
Geben Sie eine Zahl ein, um eine Aktion zu starten:
1: Neues Model trainieren.
2: Vorhandendes Model weitertrainieren.
3: Model evaluieren lassen.
4: Selber Gegen ein trainiertes Model spielen.
Modus: """)
if "1" in modus:
    modus = 1
elif "2" in modus:
    modus = 2
elif "3" in modus:
    modus = 3
else:
    modus = 4
    
if modus == 1:
    belohnungsfunktion = input("Wählen Sie mit welcher Belohnungsfunktion (B1, B2, B3) Sie trainieren wollen: ")
    
    #Umgebungen werden importiert
    KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(belohnungsfunktion)
        
    umgebung = KäsekästchenEnv(4, 4) # Trainingsumgebung wird initialisiert.
    obs, info = umgebung.reset(seed=seed)
    
    evaluierungs_umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wir initialisiert.
    evaluierungs_umgebung = Monitor(evaluierungs_umgebung, info_keywords=("is_success",))
    obs, info = evaluierungs_umgebung.reset(seed=seed)    
    
    params = input("Wenn Sie die Hyperparameter anpassen wollen, geben Sie bitte JA ein: ")
    if "JA" in params:
        lernrate = float(input("Definieren Sie die Lernrate (Standard: 0.0001, sinnvolle Range: 0.00005 - 0.001): "))
        gamma = float(input("Definieren Sie den Diskontierungsfaktor (Standard: 0.99, sinnvolle Range: 0.8 - 0.999): "))
        # Epsilon gibt an, mit welcher Wahrscheinlichkeit der Actor eine zufällige Handlung trifft.
        # Dabei beginnt Epsilon standardmässig mit dem Wert 1 und endet mit dem Wert 0.05
        # Erkundungsanteil ist dann der Parameter, der besagt, über welchen Anteil des Trainings
        # sich der Wert Epsilon von 1 zu 0.05 ändert, danach wird nur noch mit der Wahrscheinlichkeit 0.05
        # eine zufällige Handlung gewählt.
        erkundungs_anteil = float(input("Definieren Sie den Erkundungs-Anteil (Standard: 0.1, sinnvolle Range: 0.05 - 0.25): "))
        zeitschritte = int(input("Wie viele Zeitschritte wollen Sie trainieren? (Standard: 2500000, sinnvolle Range: 200000 - 3000000): "))
    else:
        # Standardwerte:
        lernrate = 0.0001
        gamma = 0.99
        erkundungs_anteil = 0.1
        zeitschritte = 2500000
        
    model = DQN(
        'MlpPolicy',
        umgebung,
        learning_rate=lernrate,
        gamma=gamma,
        exploration_fraction=erkundungs_anteil,
        buffer_size=100000,                      
        verbose=1,                               
        tensorboard_log="./Käsekästchen_Models/" 
    )
    
    # Dateinamen mit den Parametern.
    dateiname = f"{belohnungsfunktion}_lernrate{lernrate}_gamma{gamma}_erkundungs_anteil{erkundungs_anteil}_zeitschritt{zeitschritte}"
    dateiname = dateiname.replace('.', ',')
    
    # Parameter für die zyklische Evaluierung werden festgelegt.
    evaluierungs_callback = EvalCallback(evaluierungs_umgebung, best_model_save_path=f"./logs/{dateiname}",
                                         log_path="./logs/", eval_freq=10000, n_eval_episodes=100,
                                         deterministic=True, render=False)
    
    # Trainings-Metriken-Logger wird konfiguriert.
    tmp_path = f"./logs/tmp/{dateiname}/"
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    
    model.set_logger(logger) # Logger wird gesetzt.
    model.learn(total_timesteps=zeitschritte, callback=evaluierungs_callback, log_interval=4) # Das Model wird trainiert.
    model.save(dateiname) # Das fertige Model wird abgespeichert.
    print(dateiname)
    
elif modus == 2:
    model_laden = input("Geben Sie den Dateinamen des Models ein, das Sie laden wollen: ")
    
    #Umgebungen werden importiert
    KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(model_laden)
    
    umgebung = KäsekästchenEnv(4, 4)
    obs, info = umgebung.reset(seed=seed)
    
    evaluierungs_umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wir initialisiert.
    evaluierungs_umgebung = Monitor(evaluierungs_umgebung, info_keywords=("is_success",))
    obs, info = evaluierungs_umgebung.reset(seed=seed)
        
    zeitschritte = int(input("Wie viele Zeitschritte wollen Sie trainieren? "))
    
    # Dateinamen mit den Parametern.
    dateiname = f"V2_{model_laden}_V2_zeitschritt{zeitschritte}"
    dateiname = dateiname.replace('.', ',')
    
    # Parameter für die zyklische Evaluierung werden festgelegt.
    evaluierungs_callback = EvalCallback(evaluierungs_umgebung, best_model_save_path=f"./logs/{dateiname}",
                                         log_path="./logs/", eval_freq=10000, n_eval_episodes=100,
                                         deterministic=True, render=False)
    
    # Trainings-Metriken-Logger wird konfiguriert.
    tmp_path = f"./logs/tmp/{dateiname}/"
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    
    model = DQN.load(model_laden)
    model.set_env(umgebung) # Umgebung wird gesetzt.
    model.set_logger(logger) # Logger wird gesetzt.
    model.learn(
        total_timesteps=zeitschritte,
        callback=evaluierungs_callback,
        log_interval=10,
        reset_num_timesteps=False
    )
    
    model.save(dateiname) # Das fertige Model wird abgespeichert.
    print(dateiname)

elif modus == 3:
    model_laden = input("Geben Sie den Dateinamen des Models ein, das Sie laden wollen: ")
    
    # Evaluierungsumgebung wird importiert.
    from Evaluierungsumgebung import KäsekästchenEvaluierungEnv
    
    anzahl_spiele = int(input("Über wie viele Spiele wollen Sie evaluieren? (Standard: 1000): "))
        
    umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wird initialisiert.
    obs, info = umgebung.reset(seed=seed)
    umgebung = Monitor(umgebung, info_keywords=("is_success",)) 
    model = DQN.load(model_laden)
    model.set_env(umgebung)
    
    belohnung_durchschnitt, belohnung_standardabweichung = evaluate_policy(
        model, 
        umgebung, 
        n_eval_episodes=anzahl_spiele, 
        deterministic=True
    )
    
    kumulierte_belohnungen, gewinnrate = gewinnraten_evaluierung(model, umgebung, anzahl_spiele)
    
    print(f"""
    Liste der kumulierten Belohnungen jedes Spiels:
    
    {kumulierte_belohnungen}
    
    Evaluierungswerte des Model {model_laden}:
    Durchschnittliche Belohnung:  {belohnung_durchschnitt}
    Standardabweichung Belohnung: {belohnung_standardabweichung}
    Gewinnrate des Agenten: {gewinnrate: .2f})
    """) 
        
    
elif modus == 4:               
    model_laden = input("Geben Sie den Dateinamen des Models ein, das Sie laden wollen: ")
    
    #Umgebung wird importiert
    KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(model_laden)
    
    umgebung = KäsekästchenEnv(4, 4, render_mode="human")
    obs, info = umgebung.reset()
    
    model = DQN.load(model_laden)
    zeitschritte = int(input("Wie viele Zeitschritte wollen Sie spielen? "))
    for _ in range(zeitschritte):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = umgebung.step(action)
        umgebung.render() 
        time.sleep(0.5)
        if terminated or truncated:
            print("Das Spiel ist fertig. Agent-Score:", umgebung.agent_score, "Ihr Score:", umgebung.spieler2_score)
            time.sleep(5)
            obs, info = umgebung.reset()
    umgebung.close()
    
    
# Um Statistiken zu sehen: 1. in system terminal: tensorboard --logdir ./logs/tmp/ --reload_interval 30
