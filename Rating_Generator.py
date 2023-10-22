import streamlit as st
import pickle

# Loading our pre-trained model
with open('FIFA_Rating_Generator.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Creating a Streamlit app
st.title('FIFA Player Rating Prediction')
st.write('Enter player information below:')

#Features

#Position values
feature2 = st.slider('st', min_value=0.0, max_value=100.0, step=0.1)
feature3 = st.slider('rw', min_value=0.0, max_value=100.0, step=0.1)
feature4 = st.slider('rf', min_value=0.0, max_value=100.0, step=0.1)
feature5 = st.slider('cf', min_value=0.0, max_value=100.0, step=0.1)
feature6 = st.slider('lf', min_value=0.0, max_value=100.0, step=0.1)
feature7 = st.slider('lw', min_value=0.0, max_value=100.0, step=0.1)
feature8 = st.slider('rs', min_value=0.0, max_value=100.0, step=0.1)
feature9 = st.slider('ls', min_value=0.0, max_value=100.0, step=0.1)
feature10 = st.slider('cam', min_value=0.0, max_value=100.0, step=0.1)
feature11 = st.slider('lam', min_value=0.0, max_value=100.0, step=0.1)
feature12 = st.slider('ram', min_value=0.0, max_value=100.0, step=0.1)
feature13 = st.slider('rdm', min_value=0.0, max_value=100.0, step=0.1)
feature14 = st.slider('rb', min_value=0.0, max_value=100.0, step=0.1)
feature15 = st.slider('rcb', min_value=0.0, max_value=100.0, step=0.1)
feature16 = st.slider('cb', min_value=0.0, max_value=100.0, step=0.1)
feature17 = st.slider('lcb', min_value=0.0, max_value=100.0, step=0.1)
feature18 = st.slider('lb', min_value=0.0, max_value=100.0, step=0.1)
feature19 = st.slider('rwb', min_value=0.0, max_value=100.0, step=0.1)
feature20 = st.slider('cdm', min_value=0.0, max_value=100.0, step=0.1)
feature21 = st.slider('lm', min_value=0.0, max_value=100.0, step=0.1)
feature22 = st.slider('ldm', min_value=0.0, max_value=100.0, step=0.1)
feature23 = st.slider('lwb', min_value=0.0, max_value=100.0, step=0.1)
feature24 = st.slider('rm', min_value=0.0, max_value=100.0, step=0.1)
feature25 = st.slider('rcm', min_value=0.0, max_value=100.0, step=0.1)
feature26 = st.slider('cm', min_value=0.0, max_value=100.0, step=0.1)
feature27 = st.slider('lcm', min_value=0.0, max_value=100.0, step=0.1)
feature28 = st.slider('gk   ', min_value=0.0, max_value=100.0, step=0.1)

#Numeric columns
feature29 = st.slider('mentality_composure', min_value=0.0, max_value=100.0, step=0.1)
feature30 = st.slider('mentality_penalties', min_value=0.0, max_value=100.0, step=0.1)
feature31 = st.slider('mentality_vision', min_value=0.0, max_value=100.0, step=0.1)
feature32 = st.slider('mentality_positioning', min_value=0.0, max_value=100.0, step=0.1)
feature33 = st.slider('mentality_interceptions', min_value=0.0, max_value=100.0, step=0.1)
feature34 = st.slider('mentality_aggression', min_value=0.0, max_value=100.0, step=0.1)

feature35 = st.slider('pace', min_value=0.0, max_value=100.0, step=0.1)
feature36 = st.slider('physic', min_value=0.0, max_value=100.0, step=0.1)
feature37 = st.slider('defending', min_value=0.0, max_value=100.0, step=0.1)
feature38 = st.slider('dribbling', min_value=0.0, max_value=100.0, step=0.1)
feature39 = st.slider('passing', min_value=0.0, max_value=100.0, step=0.1)
feature40 = st.slider('shooting', min_value=0.0, max_value=100.0, step=0.1)

feature41 = st.slider('release_clause_eur', min_value=0.0, max_value=100.0, step=0.1)
feature42 = st.slider('international_reputation', min_value=0.0, max_value=100.0, step=0.1)
feature43 = st.slider('wage_eur ', min_value=0.0, max_value=100.0, step=0.1)
feature44 = st.slider('value_eur', min_value=0.0, max_value=100.0, step=0.1)

feature45 = st.slider('weak_foot', min_value=0.0, max_value=100.0, step=0.1)
feature46 = st.slider('age', min_value=0.0, max_value=100.0, step=0.1)
feature47 = st.slider('potential', min_value=0.0, max_value=100.0, step=0.1)

feature48 = st.slider('attacking_finishing', min_value=0.0, max_value=100.0, step=0.1)
feature49 = st.slider('attacking_heading_accuracy', min_value=0.0, max_value=100.0, step=0.1)
feature50 = st.slider('attacking_short_passing', min_value=0.0, max_value=100.0, step=0.1)
feature51 = st.slider('attacking_volleys      ', min_value=0.0, max_value=100.0, step=0.1)
feature52 = st.slider('attacking_crossing', min_value=0.0, max_value=100.0, step=0.1)

feature53 = st.slider('skill_moves', min_value=0.0, max_value=100.0, step=0.1)
feature54 = st.slider('skill_dribbling        ', min_value=0.0, max_value=100.0, step=0.1)
feature55 = st.slider('skill_curve            ', min_value=0.0, max_value=100.0, step=0.1)
feature56 = st.slider('skill_fk_accuracy           ', min_value=0.0, max_value=100.0, step=0.1)
feature57 = st.slider('skill_long_passing          ', min_value=0.0, max_value=100.0, step=0.1)
feature58 = st.slider('skill_ball_control          ', min_value=0.0, max_value=100.0, step=0.1)

feature59 = st.slider('defending_sliding_tackle', min_value=0.0, max_value=100.0, step=0.1)
feature60 = st.slider('defending_standing_tackle', min_value=0.0, max_value=100.0, step=0.1)
feature61 = st.slider('defending_marking_awareness', min_value=0.0, max_value=100.0, step=0.1)

feature62 = st.slider('movement_acceleration       ', min_value=0.0, max_value=100.0, step=0.1)
feature63 = st.slider('movement_sprint_speed       ', min_value=0.0, max_value=100.0, step=0.1)
feature64 = st.slider('movement_agility            ', min_value=0.0, max_value=100.0, step=0.1)
feature65 = st.slider('movement_reactions          ', min_value=0.0, max_value=100.0, step=0.1)

feature66 = st.slider('power_shot_power            ', min_value=0.0, max_value=100.0, step=0.1)
feature67 = st.slider('power_jumping               ', min_value=0.0, max_value=100.0, step=0.1)
feature68 = st.slider('power_stamina               ', min_value=0.0, max_value=100.0, step=0.1)
feature69 = st.slider('power_strength              ', min_value=0.0, max_value=100.0, step=0.1)
feature70 = st.slider('power_long_shots', min_value=0.0, max_value=100.0, step=0.1)


#Predicting players's Overall stat
if st.button('Predict'):
    data = [
        feature2 ,
        feature3 ,
        feature4 ,
        feature5 ,
        feature6 ,
        feature7 ,
        feature8 ,
        feature9 ,
        feature10,  
        feature11,  
        feature12,  
        feature13,  
        feature14, 
        feature15,  
        feature16, 
        feature17,  
        feature18, 
        feature19, 
        feature20,  
        feature21, 
        feature22,  
        feature23,  
        feature24,
        feature25, 
        feature26, 
        feature27,  
        feature28,    
        feature29,
        feature30,
        feature31,
        feature32,
        feature33,
        feature34, 
        feature35,   
        feature36,     
        feature37,        
        feature38,        
        feature39,      
        feature40,       
        feature41,
        feature42,
        feature43,        
        feature44,        
        feature45,        
        feature46,  
        feature47,        
        feature48,
        feature49,
        feature50,
        feature51,
        feature52,
        feature53,
        feature54,
        feature55,
        feature56,
        feature57,
        feature58,
        feature59, 
        feature60, 
        feature61,   
        feature62,     
        feature63,
        feature64,
        feature65,
        feature66,
        feature67,
        feature68,
        feature69,
        feature70 ]
        
    prediction = model.predict([data])[0]
    st.write(f'Predicted Rating: {prediction:.2f}')



