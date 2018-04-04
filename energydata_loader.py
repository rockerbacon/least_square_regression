#ignorar primeira linha (descricoes)
#ignorar primeira coluna (datas)
#ignorar segunda coluna (valor a ser estimado)
#ignorar duas ultimas colunas (valores dummy)
def openCsv (readFrom="energydata_complete.csv"):
		print("Reading dataset...\n0%")	#debug
		dataset = list(csv.reader(open(readFrom, "r"), delimiter=","))
		dataset = dataset[1:]
		shuffle(dataset)
		rows = len(dataset)
		colums = len(dataset[0])
		
		#extract correct values (y) from dataset
		y = []
		for i in range(0, rows):
			y.append(float(dataset[i][1]))
			progress = 100.0*i/(rows*(colums-3))	#debug
			print("\033[1A\r{0:.1f}%".format(progress) )	#debug
		
		#extract other values from dataset
		x = numpy.array([[0.0 for j in range(0, colums-4)] for i in range(0, rows)])
		i0 = 0
		for i1 in range(0, rows):
			j0 = 0
			for j1 in range(2, colums-2):
				x[i0][j0] = float(dataset[i1][j1])
				j0 = j0 + 1
			i0 = i0 + 1
			print("\033[1A\r{0:.1f}%".format( progress + 100.0*i1*(colums-4)/(rows*(colums-3)) ))	#debug

		print("Dataset ready")	#debug
		
		return x, y
