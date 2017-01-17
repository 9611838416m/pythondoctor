from sklearn import tree
#[age, symptom, body temp]
X = [[18,1,30],[25,2,35],[32,3,44],
		[19,4,43],[35,5,54],[54,6,67],
		[54,7,45],[64,8,47],[27,9,65],
		[24,10,25],[22,11,38]]

Y = ['fever','fev1','fev2','fev3','fev4','fev5','fev6','fev7',
		'fev8','fev9','ebola']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[34,10,53]])


print(prediction)