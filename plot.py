col=['lightskyblue','dodgerblue','blue','lightgreen','forestgreen',
     'pink','crimson','peachpuff','darkorange','cornflowerblue',
     'slateblue','xkcd:light yellow','xkcd:dandelion']

plt.subplots(figsize=(13, 5))
for i in range(len(modelchange)-1):
    plt.plot(range(modelchange[i],modelchange[i+1]),target_3[range(modelchange[i],modelchange[i+1])],color=col[integer_encoded_3[modelchange[i]]],label='Model_'+str(integer_encoded_3[modelchange[i]]))
#plt.xlim(70000,100000)
plt.ylim(0.6,2.2)
current_handles, current_labels = plt.gca().get_legend_handles_labels()

# sort or reorder the labels and handles

changed_labels = list(np.unique(current_labels))
idx=[]
for i in changed_labels:
    idx.append(np.where(np.array(current_labels)==i)[0][0])
changed_handles = list(np.array(current_handles)[np.array(idx)])
plt.legend(changed_handles,changed_labels,ncol=2,loc=(0,0.55))
#plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/chosenbaseline/zoom_chan_4.png")
plt.show()




plt.subplots(figsize=(13, 5))
plt.plot(target_3)
#plt.xlim(2200,4000)
plt.savefig("/Users/quan/Documents/Sinto_Project/SintoProject/autoencoder/result2/train_2.png")
plt.show()
