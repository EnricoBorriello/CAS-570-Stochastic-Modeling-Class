#resources:
# https://coolsymbol.com/number-symbols.html

import streamlit as st
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.stats import norm

# Set the width of the Streamlit page
st.set_page_config(layout="wide")

def mytext(text):
    return st.markdown(
        f"<p style='font-size:20px;'>{text}</p>", 
        unsafe_allow_html=True)

def hline(width):
        st.write('<hr style="border: '+str(width)+'px solid #000;">', unsafe_allow_html=True)

w = 1

# Define the tabs and their content
tabs = ['Title page',
        'Summary of Part 1',
        'Determinism & Predictability',
        'Randomness as a modeling tool',
        'Predictability of the Mean',
        'Deterministic Chaos',
        'A worked example',
        'Conclusions'
        ]

# Create a sidebar section for tab selection
selected_tab = st.sidebar.selectbox("Select a Tab", tabs)

# ---------------------------------------
# TITLE PAGE
# ---------------------------------------

if selected_tab == 'Title page':
    # Display the selected tab's content
    mytext('CAS 570 - Fundamentals of Comples adaptive Systems Science - Fall 2023')
    st.title("Determinism and Predictability")
    mytext(' ')
    mytext('Enrico Borriello')
    mytext(' ')
    st.markdown('### ❏ Their relation (or lack thereof!) ')
    st.markdown('### ❏ Randomness as a model building tool ')
    st.markdown("### ❏ Random number generators (and why they don't exist)")
    st.markdown("### ❏ Randomness & Chaos")
    st.markdown("### ❏ A worked example")


# ---------------------------------------
# Summary of Part 1
# ---------------------------------------

if selected_tab == 'Summary of Part 1':

    # Display the selected tab's content
    st.title("Summary of Part 1")


    variables = st.sidebar.checkbox("Dynamic variables")
    if variables:

        hline(w)

        st.markdown('### ❏ Dynamic variables')

        mytext('''
            ○ Their type: numerical, non-numerical
            ''')

        mytext('''
            ○ Their meaning: 
            ''')

        st.image("figures/aggregation.png", 
        #caption='caption', 
        width = 650)
 

    update = st.sidebar.checkbox("Dynamic rules")
    if update:

        hline(w)

        st.markdown('### ❏ Dynamic rules')

        mytext('''
            x (t) ⟶ x (t + Δt) ⟶ ... ⟶ iteration  ⟶ ... 
            ''')


        st.image("figures/non-linear.png", 
        #caption='caption', 
        width = 650)


        mytext('''
            No notion of linearity for non-numerical variables. (This includes Boolean variables.)
            ''')





    update = st.sidebar.checkbox("Parameters and hidden assumptions")
    if update:

        hline(w)

        st.markdown('### ❏ Parameters and hidden assumptions')

        mytext('''
            ○ Implicit assumptions on everything we don't include in our model.
            ''')

        mytext('''
            ○ This includes the meaning of our parameters.
            ''') 

        sub_option_1 = st.sidebar.checkbox("example: comp. pop. dynamics", key="sub1")
        sub_option_2 = st.sidebar.checkbox("example: SIR", key="sub2")
        sub_option_3 = st.sidebar.checkbox("example: Boids", key="sub3")

        if sub_option_1:

            st.image("figures/parameters.png", 
            #caption='caption', 
            width = 600)

        if sub_option_2:

            st.image("figures/covid.png", 
            #caption='caption', 
            width = 600)

        if sub_option_3:

            mytext('''○ Bird-oid Objects ("Boids") - Craig Reynolds 1986''')
            st.markdown("https://eater.net/boids")

            st.image("figures/boids.png", 
            #caption='caption', 
            width = 600)

# ---------------------------------------
# Determinism & Predictability
# ---------------------------------------

if selected_tab == 'Determinism & Predictability':

    # Display the selected tab's content
    st.title("Determinism & Predictability (Examples)")

    example1 = st.sidebar.checkbox("Example 1")
    if example1:

        hline(w)

        st.markdown('### ❏ Rolling a dice')

        mytext('''
            ○ Mechanical process. Fully deterministic.
            ''')

        mytext('''
            ○ Prerequisites to make predictions: 
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➊ Initial state 
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➋ Laws of mechanics
            ''')

        mytext('''
            ✱ <i>Exact prediction of the result.</i> ✱
            ''')

    example2 = st.sidebar.checkbox("Example 2")
    if example2:

        hline(w)

        st.markdown('### ❏ Weather forecast')

        mytext('''
            ○ Fully deterministic.
            ''')

        mytext('''
            ○ Prerequisites to make a prediction: 
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➊ Current state of the atmosphere 
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➋ Navier-Stokes equations of hydrodynamics
            ''')

        mytext('''
            ✱ <i> Exact forecast of the weather.<i> ✱
            ''')

        hline(w)

    example3 = st.sidebar.checkbox("Example 3")
    if example3:

        st.markdown('### ❏ Radioactive decay')

        mytext('''
            ○ Fully random process.
            ''')

        mytext('''
            ✱ <i> Too bad. It could have been used for radiometric dating.<i> ✱
            ''')

    example4 = st.sidebar.checkbox("Example 4")
    if example4:

        hline(w)

        st.markdown('### ❏ Disease Transmission')

        mytext('''
            ○ It doesn't seem deterministic, but neither does it seem random.
            ''')

        mytext('''
            ○ Prerequisites to make predictions (maybe!): 
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➊ "current state" of the population
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➋ Social dynamics
            <br>&nbsp;&nbsp;&nbsp;&nbsp;
            ➌ Physiology & pathology of the infection
            ''')

        mytext('''
            ○ Do we even know if ➋ and ➌ are more deterministic or random?
            ''')

    corrections = st.sidebar.checkbox("Corrections")
    if corrections:

        hline(w)

        mytext('''
            All ✱ <i> comments marked like this <i> ✱ 
            are both <u><strong>false and true in principle</strong></u>.
            ''')

        mytext('''
        They are rendered false by our (unavoidable) 
        <u><strong>lack of information</strong></u>
        on both the <u><strong>initial conditions</strong></u> and the 
        <u><strong>dynamical rules</strong></u>.
        ''')


# ---------------------------------------
# Randomness as a modeling tool
# ---------------------------------------

if selected_tab == 'Randomness as a modeling tool':
    st.title("Randomness as a modeling tool")

    hline(w)

    col1, col2, col3 = st.columns([2,.1,2])

    with col1:
        st.markdown('### ❏ Prerequisites for the model:')

        mytext('''
        ➊ Current state of the system: 
        <br>&nbsp;&nbsp;&nbsp;&nbsp;
        X[0]
        <br>
        ➋ Updating rule: 
        <br>&nbsp;&nbsp;&nbsp;&nbsp;
        X[0] &nbsp; ⟶ &nbsp; X[1] &nbsp; ⟶ X[2] &nbsp; ⟶ &nbsp; ...
        ''')

    with col3:
        st.markdown('### ❏ Prediction:')
        mytext('''
            ... &nbsp; ⟶ &nbsp; X[T]
            ''')
    mytext(''' 
        In any realistic scenario, 
        we have only <strong>partial information</strong> of both ➊ and ➋
        ''')

    initial_conditions = st.sidebar.checkbox('Initial conditions')
    if initial_conditions:

        hline(w)

        st.markdown('### ❏ Uncertainty on the initial conditions')
        mytext('')

        st.image("figures/deterministic_model.png", 
            #caption='caption', 
            width = 650)

        mytext('')
        mytext('')
        mytext('')
        mytext('')

        mytext(''' 
            ○ <strong>Analytic model:</strong> 
            The uncertainty on the prediction can be calculated 
            ('Propagation of uncertainty'). 
            ''')

        mytext(''' 
         ○ <strong>In any realistic case:</strong> 
         We can <u><strong>leverage randomness</strong></u> 
         as a way around the issue. 
        ''')

        mytext('')
        mytext('')
        mytext('')
        mytext('')

        st.image("figures/random_in_cond.png", 
        #caption='caption', 
        width = 650)

        mytext('')
        mytext('')
        mytext('')
        mytext('')

        mytext('ensemble of initial conditions ⟶ ensemble of predictions')

    updating_rule = st.sidebar.checkbox('Updating rule')
    if updating_rule:

        hline(w)

        st.markdown('### ❏ Uncertainty on the updating rule (Stochastic Models)')
        mytext('')

        mytext('''
            Instead of a deterministic model 
            lacking the <strong><u>details that are too difficult to model</strong></u>,
            we treat those details in a <u><strong>probabilistic</strong></u> way.
            ''')


        mytext('')
        mytext('')
        mytext('')
        mytext('')

        st.image("figures/stochastic.png", 
        #caption='caption', 
        width = 650)

        mytext('')
        mytext('')
        mytext('')
        mytext('')

        mytext('''
            We can leverage randomness to <u><strong>bypass modeling details</strong></u> of the system
            as long as the aspects we are not including produce <u><strong>compensating
            effects on average</strong></u>. 
            ''') 

    takeaway_points = st.sidebar.checkbox('Takeaway points')
    if takeaway_points:

        hline(w)

        st.markdown('### ❏ Takeaway points')
        mytext('')

        mytext('''
        ➊  Whether the "real dynamics" is deterministic 
        or not  <u><strong>matters less than we expected</strong></u>.
        Our predictions will be probabilistic in either case.
        ''')
        mytext('''
        ➋  <u><strong>Randomness can compensate for lack of information</strong></u>
        on both the present state of the system and the details
        of the dynamics.
        ''')
        mytext('''
        ➌ We need ways to generate <u><strong>random numbers</strong></u>!
        ''')

# ---------------------------------------
# Predictability of the Mean
# ---------------------------------------

if selected_tab == 'Predictability of the Mean':
    st.title("Predictability of the Mean")

    st.markdown('### ❏ Central Limit Theorem')
    mytext('')

    col1, col2, col3 = st.columns(3)

    with col1:
        mytext('Y1, Y2, ... Ym')
        mytext('''
            independent random variables
            from the same distribution
            ''')

    with col2:
        mytext('Y1 + Y2 + ... + Ym')
        mytext('''
            Gaussian random variable. 
            The larger m, the smaller the variance.
            ''')

    with col3:
            st.image("figures/gaussian.png", 
            #caption='caption', 
            width = 200)

    mytext('''
            If Y1, Y2, ... Ym are the results of our simulations,
            the're both 
            <u><strong>independent</strong></u>, and 
            <u><strong>equally distributed</strong></u>
            <br>(they're generated from the same process).
            ''')

    mytext('''
        &nbsp;&nbsp;&nbsp;&nbsp;
        mean(Y) = (Y1 + Y2 + ... + Ym)/m
        ''')

    mytext('''
            As we increase the number of simulations, 
            <u><strong>our prediction of the mean tends 
            to become determinstic</strong></u>.
            ''')
    

# ---------------------------------------
# Deterministic Chaos
# ---------------------------------------
st.set_option('deprecation.showPyplotGlobalUse', False)


if selected_tab == 'Deterministic Chaos':

    st.title("Deterministic Chaos")


    import numpy as np
    import matplotlib.pyplot as plt

    def logistic(r, x):
        return r * x * (1 - x)

    def plot_system(r, x0, n):
        fig = plt.figure(figsize=(6, 6))
        # Plot the function and the
        # y=x diagonal line.
        t = np.linspace(0, 1)
        
        
        if n > 0:
            color = 'k'
        else:
            color = 'white'
            
        plt.plot([0, 1], [0, 1], color, lw=2)
        plt.plot(t, logistic(r, t), 'k', lw=2)
        
        # Recursively apply y=f(x) and plot two lines:
        # (x, x) -> (x, y)
        # (x, y) -> (y, y)
        x = x0
        for i in range(n):
            y = logistic(r, x)
            # Plot the two lines.
            plt.plot([x, x], [x, y], 'k', lw=1)
            plt.plot([x, y], [y, y], 'k', lw=1)
            # Plot the positions with increasing
            # opacity.
            plt.plot([x], [y], 'ok', ms=10,
                    alpha=(i + 1) / n)
            x = y

        plt.xlabel('x (t)',size=16)
        plt.ylabel('x (t+1)',size=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
            
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.title(f"$r={r:.1f}, \, x_0={x0:.1f}$", size = 16)




    hline(w) 

    st.markdown('### ❏ Logistic Map')



    st.latex(r'''\boxed{\Large x_{t+1} = r\ x_t\ (1-x_t)}''')

    r = st.slider("Select the model (i.e the height of the hump)", 
        min_value=0.0, 
        max_value=4.0, 
        value=2.0, step=0.01)    

    x0 = st.slider("Select the initial condition", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.1, step=0.01)  

    col1, col2, col3, col4 = st.columns([2,10,10,2])

    with col2:
        fig = plot_system(r , x0, n = 40)
        st.pyplot(fig)

    with col3:
        T = 100
        t_series = np.linspace(0,T,T)
        x_series = [x0]
        for t in t_series[:-1]:
            x_series.append(logistic(r,x_series[-1]))

        fig = plt.figure(figsize=(6,6))
        plt.plot(t_series,x_series,'-o',c='k')
        plt.title(f"$r={r:.1f}, \, x_0={x0:.1f}$", size = 16)
        plt.xlabel('t',size=14)
        plt.ylabel('x',size=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(-1,T+1)
        plt.ylim(-0.01,1.01)
        plt.show()

        st.pyplot(fig)


    chaos = st.sidebar.checkbox('Chaos')
    if chaos:

        hline(w) 

        st.markdown('### ❏ Chaos')
        col1, col2, col3 = st.columns([1,4,1])

        with col2:

            st.image("figures/chaos.png" , 
            #caption='caption', 
            width = 950)

            mytext('')
            mytext('')
            mytext('')
            mytext('')

            mytext('''
            <i>"Chaos: When the present determines the future, 
            but the approximate present does not approximately 
            determine the future."</i>
            ''')
            mytext('''
            - Edward Lorenz (1917 – 2008)
            ''')

    points = st.sidebar.checkbox('Takeaway points')
    if points:

        hline(w)   

        st.markdown('### ❏ Takeaway points')
        mytext('')

        mytext('''
            ➊ Chaotic behaviors are highlighted (and partially dealt with)
            through <u><strong>randomization of 
            the initial state</strong></u>.
            ''')

        mytext('''
            ➋ They are characterized by 
            <u><strong>unexpectly high variation</strong></u> 
            for close initial conditions.
            ''')

        mytext('''
            ➌ The
            <u><strong>least predictive models</strong></u> 
            are not random. They are deterministic and chaotic.
            ''')

        mytext('''
        ➍ The <u><strong>upside</strong></u> 
        to chaotic systems is that they are
        <u><strong>pseudo-random number</strong></u> generators!
        ''')


    





# ---------------------------------------
# A worked example
# ---------------------------------------
 
if selected_tab == 'A worked example':
    # Display the selected tab's content
    st.title("A worked example")

    st.write("[External link](https://enricoborriello-evo-epi3-main-wsyfw4.streamlit.app/)")
    

# ---------------------------------------
# Conclusions
# ---------------------------------------

if selected_tab == 'Conclusions':

    st.title("Conclusions")



    mytext('''❏ Determinism does not imply predictability''')
    mytext('''❏ Unpredictability does not imply randomness''')
    mytext('''❏ The most unpredictable systems can be deterministic''')
    mytext('''❏ All realistic predictions are produced with uncertainty''')
    mytext('''❏ Uncertainties affect both the initial state and the dynamic rules''')
    mytext('''❏ Randomness can be used as a modeling tool to help mitigate both''')




  


# ---------------------------------------
# FOOTER
# ---------------------------------------

# Adding a space to push content upwards
st.markdown("<style>div.stButton {margin-top: 30px;}</style>", unsafe_allow_html=True)

# Adding the footer content
st.markdown(
    """
    <div style="position: fixed; bottom: 0; background-color: white; width: 100%; text-align: left;">
    Enrico Borriello -
    CAS 570 Fundamentals of CAS Science
    </div>
    """,
    unsafe_allow_html=True
)

