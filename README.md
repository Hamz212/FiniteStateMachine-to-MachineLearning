# Benchmark — Power Manager IoT 

Comparaison de 8 méthodes de décision pour remplacer la machine d'états (FSM)
du gestionnaire d'énergie d'un nœud IoT utilisé lors de tests de qualification
thermique-vide (TVCC) de satellites.



---

## Structure

```
benchmark/
├── environment.py          Simulateur physique du nœud (calibré sur le papier)
├── run_benchmark.py        Script principal
├── plot_results.py         Génération de la figure comparative
├── decision/
│   ├── base.py             Interface commune à toutes les méthodes
│   ├── fsm_baseline.py     FSM originale du papier (référence)
│   ├── fsm_optimized.py    FSM avec seuils optimisés par algorithme génétique
│   ├── random_policy.py    Politique aléatoire (sanity check)
│   ├── fuzzy_logic.py      Contrôleur flou (seuils à transitions douces)
│   ├── q_learning.py       Q-learning tabulaire (36 états, 3 actions)
│   ├── decision_tree.py    Arbre de décision distillé depuis Q-learning
│   ├── mlp_distilled.py    MLP 3→8→3 distillé depuis Q-learning
│   └── llm_distilled.py    MLP 3→16→8→3 distillé depuis un LLM  
└── results/                Créé automatiquement par run_benchmark.py
```

---

## Installation

```bash
pip install numpy matplotlib
```

---

## Utilisation

```bash
cd benchmark
python3 run_benchmark.py     # ~3 min
python3 plot_results.py
```

Résultats :
- `results/benchmark_results.csv`
- `results/benchmark_comparison.png`

---

## Méthodes comparées

| Méthode | Type | Description |
|---|---|---|
| Random Policy | aucun | Référence basse — toute méthode sensée doit faire mieux |
| FSM Baseline | règles fixes | Machine d'états du papier I2MTC 2026 |
| FSM Optimisée (GA) | optimisation | Mêmes règles, seuils optimisés par algorithme génétique |
| Logique Floue | règles douces | Fonctions d'appartenance trapézoïdales, défuzzification pondérée |
| Q-learning tabulaire | RL | Table Q discrète, 36 états, politique ε-greedy |
| Arbre de décision | supervisé (RL) | CART profondeur 6, distillé depuis Q-learning |
| MLP distillé (Q-learning) | hybride | Réseau 3→8→3 distillé depuis Q-learning |
| **MLP distillé (LLM)** | **supervisé (LLM)** | **Réseau 3→16→8→3 distillé depuis un oracle LLM** |

---

## Méthode LLM distillée — détails

La méthode `LLMDistilled` utilise un grand modèle de langage comme oracle
d'étiquetage à la place du Q-learning :

1. **Génération du dataset** : 5 000 états `{θ, Vbatt, VSC}` sont échantillonnés
   aléatoirement (avec sur-représentation des zones de bord de seuil).
2. **Étiquetage** : chaque état est soumis au LLM avec un prompt décrivant la
   physique du nœud (batterie Li-Po, supercondensateur, module ZigBee).
   Le LLM répond par l'action optimale (0, 1 ou 2).
3. **Distillation** : un MLP 3→16→8→3 est entraîné sur ces labels par
   descente de gradient (cross-entropie, 120 epochs).

Pour utiliser l'API Claude réelle, exporter la clé avant de lancer le benchmark :

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python3 run_benchmark.py
```

Sans clé, l'oracle heuristique (physique anticipatrice) est utilisé automatiquement.

---

## Métriques

- **Autonomie (j)** : durée de survie du nœud sur l'épisode de 7 jours
- **Échecs** : nombre de coupures d'alimentation (batterie épuisée)
- **TX** : nombre de transmissions ZigBee réussies
- **Perdus** : paquets de données jamais transmis
- **T_déc (µs)** : temps moyen d'une décision à l'inférence
- **Mém (B)** : empreinte mémoire RAM du modèle déployé
