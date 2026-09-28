"""Train, select using validation data, then evaluate once on held-out test data."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'smartphone_usage_synthetic.csv'
OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
FEATURES=['daily_screen_time_hours','social_media_hours','gaming_hours','phone_checks_per_day','night_use_minutes','sleep_hours','study_hours']
LABELS=['Low','Moderate','High']

def main():
    if not DATA.exists():
        raise SystemExit('Dataset missing. Run: python generate_dataset.py')
    df=pd.read_csv(DATA)
    required=set(FEATURES+['risk_level'])
    if missing:=required-set(df.columns):
        raise SystemExit(f'Missing columns: {sorted(missing)}')
    df=df.dropna(subset=['risk_level']).copy()
    for col in FEATURES:df[col]=pd.to_numeric(df[col],errors='coerce')
    if df['risk_level'].nunique()!=3 or len(df)<100:
        raise SystemExit('Need at least 100 rows and all three risk labels.')
    X=df[FEATURES];y=df['risk_level']
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.25,random_state=42,stratify=y_dev)
    models={
      'Logistic Regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000)),
      'Decision Tree':make_pipeline(SimpleImputer(strategy='median'),DecisionTreeClassifier(max_depth=7,min_samples_leaf=8,random_state=42,class_weight='balanced')),
      'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=180,max_depth=12,min_samples_leaf=3,random_state=42,class_weight='balanced',n_jobs=-1)),
    }
    scores={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pred=model.predict(X_val)
        scores[name]=float(f1_score(y_val,pred,average='macro'))
        print(f'{name}: validation macro F1 = {scores[name]:.3f}')
    best_name=max(scores,key=scores.get)
    best=models[best_name]
    best.fit(X_dev,y_dev)  # Refit on all development rows before the final test.
    pred=best.predict(X_test)
    report=classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0)
    matrix=confusion_matrix(y_test,pred,labels=LABELS)
    metrics={'data_type':'synthetic demonstration','rows':int(len(df)),'split':{'train':int(len(X_train)),'validation':int(len(X_val)),'test':int(len(X_test))},'validation_macro_f1':scores,'selected_model':best_name,'test_accuracy':float(accuracy_score(y_test,pred)),'test_macro_f1':float(f1_score(y_test,pred,average='macro')),'classification_report':report,'confusion_matrix_labels':LABELS,'confusion_matrix':matrix.tolist()}
    joblib.dump(best,OUT/'model.joblib')
    (OUT/'metadata.json').write_text(json.dumps({'features':FEATURES,'labels':LABELS,'model':best_name,'data_type':'synthetic demonstration'},indent=2))
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(f"\nSelected: {best_name}\nSynthetic test accuracy: {metrics['test_accuracy']:.3f}\nSynthetic test macro F1: {metrics['test_macro_f1']:.3f}")
    print('Confusion matrix (rows=true, columns=predicted):')
    print(pd.DataFrame(matrix,index=LABELS,columns=LABELS).to_string())
    print(f'\nSaved model and metrics in {OUT}')
if __name__=='__main__':main()
