require 'rnn'
require 'hdf5'
require 'os'
require 'nn'
require 'optim'
--require 'cutorch'
--require 'cunn'
require 'misc.attenLSTM'

--=====================================paramaters==============================================
local shuffle_trainData = false
local saveAtten = true  --whether save atten weights file or not
local outpath = "./_atten_weight"
-- model parameters
attOpt = {} 
attOpt.feat_dim = 100
attOpt.shot_num = 30

attOpt.batch_size = 200
attOpt.hidden_size = 10
attOpt.event_num = 10
attOpt.mlp_mid_dim = 5
sgd_params = {
   learningRate = 0.01,
}
local epoch_num = 10 
---------------input files
if tonumber(arg[1]) == 2 then
	print("using sequences without noise!")
	trainfile = "randseq_train.h5"
	testfile = "randseq_test.h5"
else
	print("using sequeces with noise!")
	trainfile = "randseq_train_atten.h5"
	testfile = "randseq_test_atten.h5"
end

--=============================================================================================
if not paths.dir(outpath) then
	paths.mkdir(outpath)
end

local mlp = nn.SplitTable(1)
local criterion = nn.ClassNLLCriterion()

function test (model_attenLSTM,infile_test,epoch)
	--local infile_test = "randseq_test.h5"
	print(infile_test)
	local myFile = hdf5.open(infile_test, 'r')
	local feats  = myFile:read('feature'):all() -- sample_num x shot_num(sequence) x feat_dim = 1000 x 15 x 20
	local labels = myFile:read('label'):all()
	myFile:close()
	local sample_num = feats:size(1)
	local batch_num = sample_num / attOpt.batch_size
	for i = 1, batch_num do
        	print("batch_num: " .. i)
	        local startb = 1 + (i-1)*attOpt.batch_size
        	local featbatch = feats[{{startb,startb+attOpt.batch_size-1},{},{}}] -- batch_size x shot_num x feat_dim
	        local featbatch_trans = featbatch:transpose(1,2)  -- shot_num x batch_size x feat_dim
        	--local inputs = mlp:forward(featbatch_trans)
		local inputs = featbatch_trans:clone()
	        local targets = labels[{{startb,startb+attOpt.batch_size-1}}]

	        local output,atten_weight =  unpack(model_attenLSTM:forward(inputs))
        	local loss = criterion:forward(output, targets)
	        print(loss)

		-------------save attention weight
		if saveAtten then
	                local attenfile=string.format("%s/epoch%d_from%d_batchsize%d_test.h5",outpath,epoch,startb,attOpt.batch_size)
        	        local myFile = hdf5.open(attenfile, 'w')
                	myFile:write('atten_weight', atten_weight)
	                myFile:close()
		end
	end
end

model_attenLSTM = nn.attenLSTM(attOpt)
--model_attenLSTM:cuda()
local params, gradParams = model_attenLSTM:getParameters()
--------initialize model
--[[
model_attenLSTM:clearState()
torch.save("model_init_seqLSTM.t7", model_attenLSTM)
os.exit()
--]]
--
local init_model = torch.load("model_init_seqLSTM.t7")
local init_params, init_grads = init_model:getParameters()
for i = 1, (#params)[1] do
	params[i] = init_params[i]
        gradParams[i] = init_grads[i]
end
init_model, init_params, init_grads = nil, nil, nil
--]]
params, gradParams = model_attenLSTM:getParameters()

--------------------------------test
print("\nepoch 0")
print("===========Test==============")
test(model_attenLSTM,testfile,0) 

-----------------------------main loop
local myFile = hdf5.open(trainfile, 'r')
local feats  = myFile:read('feature'):all() -- sample_num x shot_num(sequence) x feat_dim = 1000 x 15 x 20
local labels = myFile:read('label'):all() 
myFile:close()
local sample_num = feats:size(1)
local batch_num = sample_num / attOpt.batch_size 
local epoch
for epoch = 1, epoch_num do

	--------------------------------train
	print("\nepoch"..epoch)
	print("===========Train==============")
	--shuffle
	if shuffle_trainData then
		shuffle = torch.randperm(feats:size()[1]) 
	end
	--local shuffle = torch.range(1,feats:size()[1])
	for i = 1, batch_num do
		print("batch_num: " .. i)
		---------------read data
		local startb = 1 + (i-1)*attOpt.batch_size
		if shuffle_trainData then
			featbatch = torch.Tensor(attOpt.batch_size, attOpt.shot_num, attOpt.feat_dim):fill(0)
			targets = torch.Tensor(attOpt.batch_size)
			for j = 1, attOpt.batch_size do
				featbatch[j] = feats[shuffle[startb+j-1]]
				targets[j] = labels[shuffle[startb+j-1]]
			end
		else
			featbatch = feats[{{startb,startb+attOpt.batch_size-1},{},{}}] -- batch_size x shot_num x feat_dim
			targets = labels[{{startb,startb+attOpt.batch_size-1}}]
		end

		local featbatch_trans = featbatch:transpose(1,2)  -- shot_num x batch_size x feat_dim
		--local inputs = mlp:forward(featbatch_trans)
		local inputs = featbatch_trans:clone()

		--------------forward
		local outputs,atten_weight =  unpack(model_attenLSTM:forward(inputs))
		local loss = criterion:forward(outputs, targets)
		print(loss)
		-------------save attention weight
		if saveAtten then
			local attenfile=string.format("%s/epoch%d_from%d_batchsize%d.h5",outpath,epoch,startb,attOpt.batch_size)
			local myFile = hdf5.open(attenfile, 'w')
			myFile:write('atten_weight', atten_weight)
			myFile:close()
		end
		---------------backward
		local gradOutputs = criterion:backward(outputs, targets)
		model_attenLSTM:backward(inputs, gradOutputs)
		--------------update1
		--[[
		model_attenLSTM:updateParameters(0.01)
		model_attenLSTM:forget()
		model_attenLSTM:zeroGradParameters()
		--]]
		--------------update2
		--
		feval = function(params_new)
        		-- copy the weight if are changed
	        	if params ~= params_new then
        	        	params:copy(params_new)
		        end
        		return loss, gradParams
		end
		_, fs = optim.sgd(feval, params, sgd_params)
		model_attenLSTM:forget()
		model_attenLSTM:zeroGradParameters()
		--]]
		model_attenLSTM:clearState()
		collectgarbage("collect")
	end
	--------------------------------test
	print("\nepoch"..epoch)
	print("===========Test==============")
	test(model_attenLSTM,testfile,epoch) 
end
print("finish!")
