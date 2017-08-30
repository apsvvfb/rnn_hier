require 'nn'
require 'rnn'
require 'os'

local batch_size = 5
local feat_dim = 6
local hidden_size = 4
local seq_len = 10
local num = 10 
local lr = 0.01
--------initialize model
--[[
local model = nn.SeqLSTM(feat_dim,hidden_size)
--local model = nn.Sequencer(nn.Linear(feat_dim,hidden_size))
model:clearState()
torch.save("model_init_seqLSTM.t7", model)
os.exit()
--]]
local model = torch.load("model_init_seqLSTM.t7")
local model1 = torch.load("model_init_seqLSTM.t7")
local model3= torch.load("model_init_seqLSTM.t7")

--local params,  gradparams =  model:getParameters():
--local criterion = nn.SequencerCriterion(nn.MSECriterion())
-------------input, label
local input = {}
local gradOut = {}
for i = 1, num do
        x = torch.randn(seq_len,batch_size,feat_dim)
        y = torch.randn(seq_len,batch_size,hidden_size)
        table.insert(input,x)
        table.insert(gradOut,y)
end
-------------map
local map = nn.MapTable():add(model)
local out = map:forward(input)
map:backward(input,gradOut)
map:updateParameters(lr)
map:zeroGradParameters()
----------model single
loss = 0
--[[
out={}
for i = 1,num do
	out[i] = model1:forward(input[i])
end
--]]
out=model1:forward(input[num])
for i = num,1,-1 do
        gradInputs = model1:backward(input[i], gradOut[i])
	model1:updateParameters(lr)
end
model1:forget()
model1:zeroGradParameters()
----------model main(true value)
loss,gradparams_sum = 0,0
out={}
for i = num,1,-1 do
	out = model3:forward(input[i])
        gradInputs = model3:backward(input[i], gradOut[i])
	local params,  gradparams =  model3:getParameters()
	gradparams_sum = gradparams_sum + gradparams
end
model3:updateParameters(lr)
model3:forget()
model3:zeroGradParameters()

----------forward again
out = map:forward(input)
out_single,loss = {},0
for i, k in pairs(out) do
        --out1 = model1:forward(input[i])
	out3 = model3:forward(input[i])
	print(i)
	print(out3) --true value2
	print(k)    --maptable
	--print(out1) --model single
	--print(out1*2) --model single (this one is quite different from above two methods)
end
