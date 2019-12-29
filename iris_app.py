import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def main():
	st.title("Himanshu Tripathi...")

	st.title("IRIS Data Classification In Web")

	data_file = 'iris.csv'

	data_load_state = st.text("Loading data .......")

	st.header("Data Exploration")

	X = ""
	y = ""
	X_train=''
	X_test=''
	y_train=''
	y_test = ''
	y_pred = ''

	@st.cache
	def load_data():
		data = pd.read_csv(data_file)
		# st.write(data.head())
		return data

	if st.checkbox("Show Data"):
		st.write(load_data())
		data_load_state.text("Loading data....Done!")

	if st.checkbox("Show more Data showing Option"):
		select_option = st.radio("Select HEAD or TAIL", ['HEAD','TAIL'])
		if select_option == 'HEAD':
			st.write(load_data().head())
		elif select_option == "TAIL":
			st.write(load_data().tail())

	# Show Shape
	if st.checkbox("Data Shape"):
		st.write(load_data().shape)

	# Show all columns name
	if st.checkbox("Show All Columns Name"):
		st.write(load_data().columns)

	if st.checkbox("Select Dimension"):
		select_dim = st.radio("Select Row or Column", ('ROW','COLUMN'))
		if select_dim == 'ROW':
			st.write(load_data().shape[0])
		if select_dim == 'COLUMN':
			st.write(load_data().shape[1])

	if st.checkbox("Data Summary"):
		st.write(load_data().describe())

	if st.checkbox("Select Multiple Columns"):
		all_columns = load_data().columns
		names = st.multiselect("Select",all_columns)
		st.write(load_data()[names])


	st.header("Data Visualization")



	if st.checkbox("Select Type For Visualization"):
		select_type = st.radio("Select From Below",[
			'Correlation','Count Plot',
			'line plot'
			])
		if select_type == 'Count Plot':
			st.write(sns.countplot(load_data()['Species']))
			st.pyplot()
		elif select_type == 'Correlation':
			st.write(sns.heatmap(load_data().corr()))
			st.pyplot()
		elif select_type == 'line plot':
			st.write(load_data().plot(kind='bar'))
			st.pyplot()

	st.header("Now select The X values and Y values")

	if st.checkbox("Select X Columns"):
		columns = load_data().columns
		col_x = st.multiselect("select Columns X",columns)
		X = load_data()[col_x]

	if st.checkbox("Select Y Columns"):
		columns = load_data().columns
		col_y = st.multiselect("select Columns Y",columns)
		y = load_data()[col_y]
		y = y['Species'].map({
		'Iris-setosa':0,
		'Iris-versicolor':1,
		'Iris-virginica':2
		})




	if st.checkbox("Show X values and y Values"):
		st.write("X Values")
		st.write(X)
		st.write("Y values")
		st.write(y)

	#sklearn packags
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import confusion_matrix, accuracy_score

	sc = StandardScaler()

	st.header("Split DataSet into Train and Test")
	if st.checkbox("Split"):
		X_train,X_test,y_train,y_test = train_test_split(
			X,y,test_size=0.2,random_state=20)
		

	if st.checkbox("Preform Standard Scale"):
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)

	if st.checkbox("Show X_test,X_train,y_test,y_train"):
		st.write("X_train")
		st.write(X_train)
		st.write(X_train.shape)
		st.write("X_test")
		st.write(X_test)
		st.write(X_test.shape)
		st.write("y_train")
		st.write(y_train)
		st.write(y_train.shape)
		st.write("y_test")
		st.write(y_test)
		st.write(y_test.shape)

	st.header("Not its time to preform ML Algo. I'm going to use Logestic Regression")	

	clf = LogisticRegression(random_state=0)
	if st.checkbox("Fit Data"):
		clf.fit(X_train,y_train)
		st.success("Fit Successfully")

	if st.checkbox("Predict"):
		y_pred = clf.predict(X_test)
		st.success("Predict Successfully")

	st.header("Now its time to see the Accuracy")

	if st.checkbox("Show Accuracy"):
		st.error(accuracy_score(y_test,y_pred)*100)

	st.header("Confusion matrix")

	if st.checkbox("Confusion matrix"):
		cm = confusion_matrix(y_test,y_pred)
		st.write(sns.heatmap(cm,annot=True))
		st.pyplot()


	st.title("Thanks for Watching......")
	st.header("Himanshu Tripathi")

if __name__ == '__main__':
	main()