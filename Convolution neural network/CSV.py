#存EXCEL(Validation)
#validation_data=(x_test, y_test)

#model.save(SavePath + "weights.hdf5")
model.load_weights(SavePath + "_")
#model.save('cnn_NP_vs_P.h5')

scores = model.evaluate(x_Test, y_Test, verbose=1)

out = model.predict(x_Test)

file = open(SavePath + '_.csv', 'w', newline='')
csvCursor = csv.writer(file)


d0 = ["", "loss",scores[0],"accuracy",scores[1]]
csvCursor.writerow(d0)


# write header to csv file
csvHeader = ['', 'test0', 'test1', 'true0', 'true1']
csvCursor.writerow(csvHeader)



#csvCursor.writerows(data_filepathTS)
sum = len(y_Test)

z0 = 0#tpnew_test_y
z1 = 0#fn
z2 = 0#fp
z3 = 0#tn
for x in range(sum):
    if(y_Test[x,0] < y_Test[x,1]) and (out[x,0] < out[x,1]):
        z0 = z0 + 1
for x1 in range(sum):
    if(y_Test[x1,0] < y_Test[x1,1]) and (out[x1,1] < out[x1,0]):
        z1 = z1 + 1
for x2 in range(sum):
    if(y_Test[x2,1] < y_Test[x2,0]) and (out[x2,0] < out[x2,1]):
        z2 = z2 + 1
for x3 in range(sum):
    if(y_Test[x3,1] < y_Test[x3,0]) and (out[x3,1] < out[x3,0]):
        z3 = z3 + 1
d1 = ["", "TP",z0,"FN",z1,"FP",z2,"TN",z3]
d2 = ["", "sensitivity",z0/(z0+z1),"specificity",z3/(z3+z2),"precision",z0/(z0+z2),"F1",(2*z0/(z0+z2)*z0/(z0+z1))/(z0/(z0+z1)+z0/(z0+z2))]
csvCursor.writerow(d1)
csvCursor.writerow(d2)

# generate a random data
for i in range(sum):
    data = [test_savepath[i],out[i,0], out[i,1],y_Test[i,0], y_Test[i,1]]
    csvCursor.writerow(data)

