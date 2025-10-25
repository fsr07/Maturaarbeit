# Einzelne Codezeilen übernommen aus der offiziellen Stable-Baselines3 Dokumentation:
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
# https://stable-baselines3.readthedocs.io/en/master/common/monitor.html#module-stable_baselines3.common.monitor
# Lizenz: MIT License (https://opensource.org/licenses/MIT)

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import os


seed = 0 # Durch einen festen Seedwert wird die Reproduzierbarkeit sichergestellt.


# Rückrufs-Funktion, damit die Gewinnrate berechnet wird.
def gewinnrate_evaluierung(modell, umgebung, anzahl_spiele):
    global seed
    siege = 0
    spielbelohnung_liste = []
    
    for _ in range(anzahl_spiele):
        obs, info = umgebung.reset(seed=seed+_)
        fertig = False
        spielbelohnung = 0
        
        while not fertig:
            action, _ = modell.predict(obs, deterministic=True)
            obs, reward, fertig, truncated, info = umgebung.step(action)
            spielbelohnung += reward

            if fertig:
                spielbelohnung_liste.append(spielbelohnung)
                if info['is_success']:
                    siege += 1

    gewinnrate = siege / anzahl_spiele
    
    return spielbelohnung_liste, gewinnrate


# Entsprechende Umgebung wird importiert.
def umgebung_importieren(belohnungsfunktion):
    if "B2" in belohnungsfunktion:
        from Belohnungsfunktion_2 import KäsekästchenEnv
    elif "B3" in belohnungsfunktion:
        from Belohnungsfunktion_3 import KäsekästchenEnv
    elif "B1" in belohnungsfunktion:
        from Belohnungsfunktion_1 import KäsekästchenEnv
    else:
        from Belohnungsfunktion_1 import KäsekästchenEnv
        print("Als Umgebung wurde automatisch B1 gewählt.")
    
    # Evaluierungsumgebung wird importiert.
    from Evaluierungsumgebung import KäsekästchenEvaluierungEnv
    
    return KäsekästchenEnv, KäsekästchenEvaluierungEnv


# Funktion, um Nutzer nach Modus zu fragen.
def user_abfrage():
    os.system('cls' if os.name == 'nt' else 'clear')
    # Printet Programmnamen in ASCII-Art
    
    print("""
 /$$   /$$   /$ /$                       /$$         /$ /$               /$$               /$$                          
| $$  /$$/  |_/|_/                      | $$        |_/|_/              | $$              | $$                          
| $$ /$$/   /$$$$$$   /$$$$$$$  /$$$$$$ | $$   /$$  /$$$$$$   /$$$$$$$ /$$$$$$    /$$$$$$$| $$$$$$$   /$$$$$$  /$$$$$$$ 
| $$$$$/   |____  $$ /$$_____/ /$$__  $$| $$  /$$/ |____  $$ /$$_____/|_  $$_/   /$$_____/| $$__  $$ /$$__  $$| $$__  $$
| $$  $$    /$$$$$$$|  $$$$$$ | $$$$$$$$| $$$$$$/   /$$$$$$$|  $$$$$$   | $$    | $$      | $$  \ $$| $$$$$$$$| $$  \ $$
| $$\  $$  /$$__  $$ \____  $$| $$_____/| $$_  $$  /$$__  $$ \____  $$  | $$ /$$| $$      | $$  | $$| $$_____/| $$  | $$
| $$ \  $$|  $$$$$$$ /$$$$$$$/|  $$$$$$$| $$ \  $$|  $$$$$$$ /$$$$$$$/  |  $$$$/|  $$$$$$$| $$  | $$|  $$$$$$$| $$  | $$
|__/  \__/ \_______/|_______/  \_______/|__/  \__/ \_______/|_______/    \___/   \_______/|__/  |__/ \_______/|__/  |__/
                                                                                                                        
                                                                                                                        
                                                                                                                        
 /$$                       /$$$$$$$$ /$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$                                             
| $$                      | $$_____//$$__  $$| $$__  $$ /$$$_  $$|_____ $$/                                             
| $$$$$$$  /$$   /$$      | $$     | $$  \__/| $$  \ $$| $$$$\ $$     /$$/                                              
| $$__  $$| $$  | $$      | $$$$$  |  $$$$$$ | $$$$$$$/| $$ $$ $$    /$$/                                               
| $$  \ $$| $$  | $$      | $$__/   \____  $$| $$__  $$| $$\ $$$$   /$$/                                                
| $$  | $$| $$  | $$      | $$      /$$  \ $$| $$  \ $$| $$ \ $$$  /$$/                                                 
| $$$$$$$/|  $$$$$$$      | $$     |  $$$$$$/| $$  | $$|  $$$$$$/ /$$/                                                  
|_______/  \____  $$      |__/      \______/ |__/  |__/ \______/ |__/                                                   
           /$$  | $$                                                                                                    
          |  $$$$$$/                                                                                                    
           \______/                                                                                                     
""")
    print("=" * 120)
    print("Steuerungsprogramm des DQN-Trainings von Käsekästchen".center(120))
    print("=" * 120)
    print("")
    print("─" * 80)
    print("HAUPTMENÜ".center(80))
    print("─" * 80)
    print("")

    
    print("""
Geben Sie eine Zahl ein, um eine Aktion zu starten:
[1] Neues Modell trainieren.
[2] Vorhandendes Modell weitertrainieren.
[3] Modell evaluieren lassen.
[4] Selber Gegen ein trainiertes Modell spielen.""")
    print("")
    print("─" * 80)
    modus = input("Ihre Wahl: ")
    print("")
    print("─" * 80)
    if "1" in modus:
        modus_1()
    elif "2" in modus:
        modus_2()
    elif "3" in modus:
        modus_3()
    elif "4" in modus:
        modus_4()
    else:
        print("Unzulässige Eingabe")
        user_abfrage()


# Mit Modus 1 können neue Modelle trainiert werden. 
def modus_1():
    global seed
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("")
    print("═" * 80)
    print("NEUES MODELL TRAINIEREN".center(80))
    print("═" * 80)
    print("")
    
    belohnungsfunktion = input("Wählen Sie mit welcher Belohnungsfunktion (B1, B2, B3) Sie trainieren wollen: ")
    
    # Umgebungen werden importiert
    KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(belohnungsfunktion)
        
    umgebung = KäsekästchenEnv(4, 4) # Trainingsumgebung wird initialisiert.
    obs, info = umgebung.reset(seed=seed)
    
    evaluierungs_umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wir initialisiert.
    evaluierungs_umgebung = Monitor(evaluierungs_umgebung, info_keywords=("is_success",))
    obs, info = evaluierungs_umgebung.reset(seed=seed)
    
    print("")
    print("─" * 80)
    
    parameter = input("Hyperparameter anpassen? [1=Ja / 0=Nein]: ")
    if "1" in parameter:
        try:
            print("")
            print("─" * 33 +  "Hyperparameter" + "─" * 33)
            lernrate = float(input("Lernrate (Standard: 0.0001, Range: 0.00005 - 0.001): "))
            gamma = float(input("Diskontierungsfaktor (Standard: 0.99, Range: 0.8 - 0.999): "))
            explorationsphasenanteil = float(input("Explorationsphasenanteil (Standard: 0.1, Range: 0.05 - 0.25): "))
            zeitschritte = int(input("Zeitschritte (Standard: 2500000, Range: 200000 - 3000000): "))
            print("─" * 80)
            print("")
        except ValueError:
            
            print("")
            print("Unzulässige Eingabe")
            time.sleep(2)
            return modus_1()
    else:
        print("")
        print("Es werden die Standardwerte verwendet")
        # Standardwerte:
        lernrate = 0.0001
        gamma = 0.99
        explorationsphasenanteil = 0.1
        zeitschritte = 2500000
        
    modell = DQN(
        'MlpPolicy',
        umgebung,
        learning_rate=lernrate,
        gamma=gamma,
        exploration_fraction=explorationsphasenanteil,
        buffer_size=100000,                      
        verbose=1,                               
        tensorboard_log="./Käsekästchen_Models/" 
    )
    
    # Dateinamen mit den Parametern.
    dateiname = f"{belohnungsfunktion}_lernrate{lernrate}_gamma{gamma}_explorationsphasenanteil{explorationsphasenanteil}_zeitschritt{zeitschritte}"
    dateiname = dateiname.replace('.', ',')
    
    # Parameter für die zyklische Evaluierung werden festgelegt.
    evaluierungs_callback = EvalCallback(evaluierungs_umgebung, best_model_save_path=f"./logs/{dateiname}",
                                         log_path="./logs/", eval_freq=10000, n_eval_episodes=100,
                                         deterministic=True, render=False)
    
    # Trainings-Metriken-Logger wird konfiguriert.
    tmp_path = f"./logs/tmp/{dateiname}/"
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    
    modell.set_logger(logger) # Logger wird gesetzt.
    
    print("")
    print("─" * 28 + "ANLEITUNG ZU TENSORBOARD" + "─" * 28)
    input("""
1. Öffnen Sie den Pfad dieser Datei im Windows-Explorer
2. Tippen Sie in die Adressleiste: cmd
3. Im Terminal eingeben: tensorboard --logdir ./logs/tmp/ --reload_interval 30 
""")
    print("─" * 80)
    print("")
    input("Drücken Sie ENTER um das Training zu starten...")
    
    
    modell.learn(total_timesteps=zeitschritte, callback=evaluierungs_callback, log_interval=4) # Das Modell wird trainiert.
    modell.save(dateiname) # Das fertige Modell wird abgespeichert.
    print("Modell wurde gespeichert: " + dateiname)
    
    time.sleep(5)
    user_abfrage()
    
    
# Mit Modus 2 können bereits trainierte Modelle noch mehr trainiert werden.    
def modus_2():
    global seed
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("")
    print("═" * 80)
    print("MODELL WEITERTRAINIEREN".center(80))
    print("═" * 80)
    print("")
    
    try:
        modell_laden = input("Dateiname des Modells, das Sie laden wollen: ")
        #Umgebungen werden importiert
        KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(modell_laden)
        
        umgebung = KäsekästchenEnv(4, 4)
        obs, info = umgebung.reset(seed=seed)
        
        evaluierungs_umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wir initialisiert.
        evaluierungs_umgebung = Monitor(evaluierungs_umgebung, info_keywords=("is_success",))
        obs, info = evaluierungs_umgebung.reset(seed=seed)
    except:
        print("Unzulässige Eingabe")
        return modus_2()
    
    
    try:
        print("")
        print("─" * 80)
        zeitschritte = int(input("Wie viele Zeitschritte wollen Sie trainieren? "))
        print("")
    except ValueError:
        print("Unzulässige Eingabe")
        time.sleep(2)
        return modus_2()
    
    # Dateinamen mit den Parametern.
    dateiname = f"V2_{modell_laden}_V2_zeitschritt{zeitschritte}"
    dateiname = dateiname.replace('.', ',')
    
    # Parameter für die zyklische Evaluierung werden festgelegt.
    evaluierungs_callback = EvalCallback(evaluierungs_umgebung, best_model_save_path=f"./logs/{dateiname}",
                                         log_path="./logs/", eval_freq=10000, n_eval_episodes=100,
                                         deterministic=True, render=False)
    
    # Trainings-Metriken-Logger wird konfiguriert.
    tmp_path = f"./logs/tmp/{dateiname}/"
    logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    
    modell = DQN.load(modell_laden) # Modell wird geladen.
    modell.set_env(umgebung) # Umgebung wird gesetzt.
    modell.set_logger(logger) # Logger wird gesetzt.
    
    print("")
    print("─" * 28 + "ANLEITUNG ZU TENSORBOARD" + "─" * 28)
    input("""
1. Öffnen Sie den Pfad dieser Datei im Windows-Explorer
2. Tippen Sie in die Adressleiste: cmd
3. Im Terminal eingeben: tensorboard --logdir ./logs/tmp/ --reload_interval 30 
""")
    print("─" * 80)
    print("")
    input("Drücken Sie ENTER um das Training zu starten...")
    
    modell.learn( # Modell beginnt zu lernen.
        total_timesteps=zeitschritte,
        callback=evaluierungs_callback,
        log_interval=10,
        reset_num_timesteps=False
    )
    
    modell.save(dateiname) # Das fertige Modell wird abgespeichert.
    print("Modell wurde gespeichert: " + dateiname)
    print("")
    input("Drücken Sie ENTER um fortzufahren...")
    
    user_abfrage()


# Mit Modus 3 können trainierte Modelle in der Evaluierungsumgebung final evaluiert werden.
def modus_3():
    global seed
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("")
    print("═" * 80)
    print("MODELL EVALUIEREN".center(80))
    print("═" * 80)
    print("")
    
    try:
        modell_laden = input("Dateiname des Modells, das Sie laden wollen: ")
        modell = DQN.load(modell_laden)
        anzahl_spiele = int(input("Anzahl Evaluierungsspiele (Standard: 1000): "))
        print("")
    except:
        print("Unzulässige Eingabe")
        time.sleep(2)
        return modus_3()

    # Evaluierungsumgebung wird importiert.
    from Evaluierungsumgebung import KäsekästchenEvaluierungEnv
    umgebung = KäsekästchenEvaluierungEnv(4, 4) # Evaluierungsumgebung wird initialisiert.
    obs, info = umgebung.reset(seed=seed)
    umgebung = Monitor(umgebung, info_keywords=("is_success",)) # Trackt die Anzahl der Gewinne bei der Evaluierung für gewinnrate_evaluierung()
    modell.set_env(umgebung)
    
    belohnung_durchschnitt, belohnung_standardabweichung = evaluate_policy(
        modell, 
        umgebung, 
        n_eval_episodes=anzahl_spiele, 
        deterministic=True
    )
    
    # Die mit evaluate_policy die Gewinnrate nicht getrackt werden kann, wurde dazu eine extra Fuktion programmiert.
    # Diese Funktion evaluiert jedoch die Gewinnrate separat von evaluate_policy
    kumulierte_belohnungen, gewinnrate = gewinnrate_evaluierung(modell, umgebung, anzahl_spiele)
    
    print("")
    print("─" * 29 + "EVALUIERUNGSERGEBNISSE" + "─" * 29)
    print(f"""
Evaluierungswerte des Modell {modell_laden}:
Durchschnittliche Belohnung:  {belohnung_durchschnitt}
Standardabweichung Belohnung: {belohnung_standardabweichung}
Gewinnrate des Agenten: {gewinnrate}
""")
    belohnungen_liste =input("Wollen Sie die Liste aller Belohnungen [1=Ja / 0=Nein] ")
    if belohnungen_liste == "1":
        print("""
Liste der kumulierten Belohnungen jedes Spiels:
{kumulierte_belohnungen}
""")
    print("─" * 80)
    print("")
    input("Drücken Sie ENTER um fortzufahren...")
    user_abfrage()
        

# Mit Modus 3 kann man selbst gegen ein trainiertes Modell spielen.
def modus_4():
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("")
    print("═" * 80)
    print("GEGEN MODELL SPIELEN".center(80))
    print("═" * 80)
    print("")
    
    try:
        modell_laden = input("Dateiname des Modells, das Sie laden wollen: ")
        #Umgebung wird importiert
        KäsekästchenEnv, KäsekästchenEvaluierungEnv = umgebung_importieren(modell_laden)
        zeitschritte = int(input("Anzahl Zeitschritte: "))
        umgebung = KäsekästchenEnv(4, 4, render_mode="human")
        obs, info = umgebung.reset()
        
        modell = DQN.load(modell_laden)
    except:
        print("Unzulässige Eingabe")
        time.sleep(2)
        return modus_4()
    
    for _ in range(zeitschritte):
        action, _ = modell.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = umgebung.step(action)
        umgebung.render() 
        time.sleep(0.5)
        if terminated or truncated:
            print("")
            print("─" * 40)
            print("Spiel beendet!")
            print(f"Agent-Score: {umgebung.agent_score}")
            print(f"Ihr Score:   {umgebung.spieler2_score}")
            print("─" * 40)
            print("")
            time.sleep(2)
            obs, info = umgebung.reset()
    umgebung.close()
    
    input("Drücken Sie ENTER um fortzufahren...")
    
    user_abfrage()


# Programm wird gestartet
user_abfrage()