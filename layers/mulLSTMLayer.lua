require 'rnn'
require 'math'
require 'nn'
require 'mulLSTM'
--require 'cutorch'
--require 'cunn'
local mulLSTMLayer, Parent = torch.class('nn.mulLSTMs', 'nn.Module')

function mulLSTMLayer:__init(opt)
	Parent.__init(self)
        self.feat_dim = utils.getopt(opt, 'feat_dim')
        self.hidden_size = utils.getopt(opt, 'hidden_size')
        self.shot_len = utils.getopt(opt, 'shot_len')
	self.shot_num = utils.getopt(opt, 'shot_num')
        self.event_num =  utils.getopt(opt, 'event_num')
        self.batch_size = utils.getopt(opt, 'batch_size')
	self.overlap = utils.getopt(opt, 'overlap')

	--self.mulLSTMmodel = nn.mulLSTM(self)
	self.mulLSTMmodel = nn.MapTable():add(nn.mulLSTM(self))
	
	self.start_idxs = torch.Tensor(self.shot_num):fill(1)
	self.end_idxs = torch.Tensor(self.shot_num):fill(self.shot_len)
	for i = 2, self.shot_num do
		self.start_idxs[i] = self.end_idxs[i-1] - self.overlap + 1
		self.end_idxs[i] = self.start_idxs[i] + self.shot_len -1
	end
end

function mulLSTMLayer:parameters()
        local p1,g1 = self.mulLSTMmodel:parameters()

        local params = {}
        for k,v in pairs(p1) do table.insert(params, v) end

        local grad_params = {}
        for k,v in pairs(g1) do table.insert(grad_params, v) end

        return params, grad_params
end

function mulLSTMLayer:updateOutput(input)
	--input:	seq_len x batch_size x feat_dim
	--output:	shot_num x batch_size x hidden_dim
	if input:type() == 'torch.CudaTensor' then
                --d_hidden_ = d_hidden_:cuda()
		self.model:cuda()
        end
	local seq_len = #input[1]
	--local output = torch.Tensor(self.shot_num, self.batch_size, self.hidden_size)
	local start_idx, end_idx
	self.input_singles = {}
	for i = 1, self.shot_num do
		local input_single=torch.Tensor(self.shot_len, self.batch_size, self.feat_dim):fill(0)
		start_idx = self.start_idxs[i] 
		end_idx = self.end_idxs[i] 
		if end_idx < seq_len then
			input_single = input[{ {start_idx,end_idx},{},{} }]
		elseif start_idx < seq_len and end_idx > seq_len then
			---------padding zero in right
			--input_single[{ {1,seq_len-start_idx+1},{},{} }] = input[{ {start_idx,seq_len},{},{} }]
			---------padding zero in left
			input_single[{ {self.shot_len-seq_len+start_idx,self.shot_len},{},{} }] = input[{ {start_idx,seq_len},{},{} }]
		end
		
		--input and output of mulLSTMmodel:
		--	input: shot_len x batchsize x feat_dim
	        --	output: batchsize x hidden_size
		table.insert(self.input_singles,input_single)
	end
	output_table = self.mulLSTMmodel:forward(self.input_singles)
	output = nn.Reshape(self.shot_num, self.batch_size, self.hidden_size)(nn.JoinTable(1):forward(output_table))

	return output
end

function mulLSTMLayer:updateGradInput(input, gradOutput)
	gradOutput_table = nn.SplitTable(1)(gradOutput)	
	local gradInput_table = self.mulLSTMmodel:backward(self.input_singles,gradOutput_table)
	self.gradInput = torch.Tensor(self.seq_len, self.batch_size, self.feat_dim):fill(0)
	for i = 1, self.shot_num do
		if self.start_idxs[i] <= self.seq_len then
                        local end_idx = math.min(self.end_idxs[i],self.seq_len)
			self.gradInput[{ {self.start_idxs[i],end_idx},{},{} }] = gradInput_table[i]
                end
	end
	--self.mulLSTMmodel:forget()
	return self.gradInput
end
