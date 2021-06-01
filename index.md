## Welcome
I am Jan Pastorek and this is my webpage for bachelor thesis.

### Name of an assignment
Machine learning for nonlocal games

### Supervisor
Mgr. Daniel Nagaj, PhD.

### Anotation
Nonlocal games are a key concept in quantum information, utilized from complexity theory to certification of quantum devices. They involve two or more players that win if they provide properly correlated answers to questions. The typical example is the CHSH game, related to Bell inequalities that can be violated by quantum mechanics. In this game, quantum players have a higher winning probability than classical players. Actually determining the optimal winning probability is a difficult problem in general. In this thesis, the student will investigate a variety of nonlocal games and search for optimal quantum strategies with the help of machine learning strategies (reinforcement learning).


### Goal
Optimalization of quantum strategies for non-local CHSH games via machine learning and evolutionary algorithms.

### Code
see readme at  <a href="https://github.com/janpastorek/Bachelor-Thesis">Project</a> and download project files
<a href="https://github.com/janpastorek/Bachelor-Thesis/archive/refs/heads/master.zip">here</a>

### Bachelor thesis

<a href="Bachelor_s_Thesis (5).pdf">Download File</a>

### Structure of text - Main chapters


<ol>
<li>Introduction  (What is the problem? Why it is important and interesting?)
</li>
<li>Quantum Mechanics </li>
<li>Non local games ( problem in details)</li>
<li>Reinforcement machine learning  (How are we going to solve this? -- Methods)</li>
<li>Genetic algorithms</li>
<li>Simulated annealing</li>
<li>Implementation (How have we solved this?)</li>
 <li>Results</li>
 <li>Discussion and conclusion</li>
</ol>

### Diary (in Slovak)

<details>
<summary>Click and see!</summary>
<ul>
<li> 16.2 - 23.2 </li>
<ul>
<li> Implementoval som GPU tensorflow model do trénovania môjho Reinforcement Agenta.</li>
<li> Pracoval som na funkcii, ktorá porovnáva najlepšiu klasickú a najlepšiu kvantovú taktiku. A vyberie také hry, ktoré majú najväčšie rozdiely.</li>
<li> Pridal som nové actions, ktoré vie vykonávať agent. (spomaľ, zrýchli) </li>
<li> Refactoring a väčšia abstrakcia hrier, genetických algoritmov etc. </li>
</ul>
<li> 23.2 - 2.3 </li>
<ul>
<li> Implementoval som PyTorch Deep Reinforcement DQN Agenta</li>
<li> Pracoval som na funkcii, ktorá porovnáva najlepšiu klasickú a najlepšiu kvantovú taktiku. A vyberie také hry, ktoré majú najväčšie rozdiely - upravil som ju, aby fungovala spravne.</li>
<li> Stretol som sa so skolitelom - urcili sme si uz finalne ciele mojej bakalarky</li>
<li> Refactoring. </li>
<li> Pracoval som na kapitole NonLocal games v LaTeXu</li>
</ul>
<li> 2.3 - 9.3 </li>
<ul>
<li> Snazil som sa optimalizovat DQN agenta</li>
<li> Implementoval som databazu, do ktorej sa budu ukladat uz preskumane hry, a ak sa znovu preskumaju, tak upsert ak sa najde lepsia hodnota.</li>
<li> Stretol som sa so skolitelom - zhodnotili sme tohtotyzdnovu pracu</li>
<li> Refactoring v triede Environment. </li>
<li> Urobil som state diagram, ako sa uci reinforcement learning, ako je to strukturovane </li>
<li> Viacere parametre som povytiahol von, nech si to pouzivatel moze sam nastavit </li>
<li> Pracoval som na kapitole NonLocal games v LaTeXe</li>
<li> Pracoval som na kapitole Reinforcement learning v LaTeXe</li>
</ul>
<li> 9.3 - 16.3 </li>
<ul>
<li> Snazil som sa optimalizovat DQN agenta, jeho ucenia pomocou memoizacie </li>
<li> Doplnil som nove stlpce do databazy (predtym som si tam neukladal cestu, ktoru sa naucil)</li>
<li> Stretol som sa so skolitelom - zhodnotili sme tohtotyzdnovu pracu</li>
<li> Refactoring nad v podstate vsetkymi triedami. </li>
<li> Doplnil som state diagram, ktory ukazuje co robi Environemnt v jednom kroku (sluzi to nato, aby bolo vidno, ze jeden hrac je od druheho oddeleny) </li>
<li> Podarilo sa mi rozbehat PyTorch na GPU </li>
<li> Zacal som implementovat Complex PyTorch siet (lebo kvantovy stav moze byt aj komplexny) </li>
<li> Odskusal som si optimalizaciu hyperparametrov pomocou genetic algoritmu </li>
<li> Pridal som zopar dalsich testov </li>
<li> Napisal som si viacero reward funkcii a tie tiez som zahrnul ako parameter do genetic algoritmu (aby pripadne vybral tu optimalznu z nich) </li>
<li> Pracoval som na kapitole Reinforcement learning v LaTeXe</li>
</ul>
<li> 16.3 - 23.3 </li>
<ul>
<li> Stretol som sa s panom Petrovicom a radil som sa s nim ohladom reinforcement learningu</li>
<li> Implementoval som annealing pri vybere bran </li>
<li> Stretol som sa so skolitelom - zhodnotili sme tohtotyzdnovu pracu, </li>
<li> Refactoring. </li>
<li> Zacal som implementovat Complex PyTorch siet (lebo kvantovy stav moze byt aj komplexny), ale nakoniec teda rozlozim complexny vektor na 2*taky dlhy realnych cisiel </li>
<li> Pracoval som na kapitole Reinforcement learning a Nonlocal games v LaTeXe</li>
<li> Spracoval som prezentáciu pre svoju bakalársku prácu </li>
</ul>
<li> 23.3 - 30.3 </li>
<ul>
<li> Refactoring </li>
<li> Nechal som vypocty ist 1 den, a vysledky som spracoval a poslal skolitelovi </li>
<li> Pracoval som na kapitole Reinforcement learning a Nonlocal games v LaTeXe</li>
</ul>
      <li> 30.3 - 6.4 </li>
<ul>
<li> Refactoring </li>
<li> Prvotná analýza výsledkov. </li>
<li> Testovanie hypotéz </li>
</ul>
      <li> 6.4 - 13.4 </li>
<ul>
<li> Refactoring </li>
<li> Škálovanie hry pre 2xEPR páry pre XOR paralelne CHSH hry </li>
<li> Študoval som teóriu zložitostí, nelokálne hry = trieda zložitostí MIP*, pretože mám o tom jednu podkapitolu </li>
<li> Dopísal som kapitoly Nonlocal games a Reinforcement learning </li>
</ul>
      <li> 13.4 - 20.4 </li>
<ul>
<li> Refactoring a dokumentácia kódu </li>
<li> Postupne opisujem implementaciu </li>
</ul>
            <li> 20.4 - 27.4 </li>
<ul>
<li> Refactoring a dokumentácia kódu, už som skončil implementáciu </li>
<li> Postupne opisujem implementaciu, robím obrázky a diagramy, ako u mna prebieha vypocet a ako na seba tie algoritmy nadvazuju, tiez ako si reprezentujem dolezite hodnoty, co si ukladam a opisujem aj dolezite metody pre tento problem </li>
</ul>
            <li> 27.4 - 4.5 </li>
<ul>
<li> Refactoring a dokumentácia kódu </li>
<li> Doopisoval som implementaciu, uz zapracovavam skolitelove komentare a este pridavam nejake grafy do vysledkov, ktore opisujem </li>
<li> Už robím na závere bakalárskej práce </li>
</ul>
                  <li> 4.5 - 11.5 </li>
<ul>
<li> Refactoring, dokumentácia kódu (uz som si vygeneroval dokumentaciu z kodu) a doladzovanie textu </li>
</ul>
</ul>
</details>

### Deadlines, Milestones
<ul>
<li> web page (11:30 29.10.2020) -> Done on time </li>

<li> set milestones and deadlines (11:30 10.11.2020)  -> Done on time </li>

<li> collecting sources(11:30 01.12.2020) -> Done on time </li>

<li> 10 pages and prototype (15.1.2021) -> Done on time </li>
  
<li> final program (15.3.2021) -> Postponed till (31.3.2021) </li>

<li> completing main part of bachelor thesis (15.4.2021)  -> Done on time </li>

<li> final testing and submitting (15.5.2021) -> Done on time </li>

</ul>

### Sources

<a href="https://docs.google.com/document/d/1qOo0u9xQUziNXVe9xGa9vmwTgQ-spIhnZ7xd4Q12jvs/edit?usp=sharing">References</a>

<a href="https://www.canva.com/design/DAEPEqLIsWM/ij-WJ0Wpchf-UAXgLVFSWA/view?utm_content=DAEPEqLIsWM&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton">Presentation of sources, and, plan and progress of work</a>
