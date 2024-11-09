import pandas as pd

def people_detected_over_time(objects_per_frame, fps):
    df = pd.DataFrame(objects_per_frame, columns=['frame', 'people'])
    df['second'] = df['frame'] // fps 
    people_per_second = df.groupby('second')['people'].mean().reset_index()
    return people_per_second