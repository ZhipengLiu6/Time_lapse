--

function getdata(datapath, batch)
	local  dataset = {}
	print("Loading data ......")
	local data = torch.load(datapath)
	print("Load data done!")
	local nsamples = data:size(1)
	local nchannel = data:size(2)
	local nrows = data:size(3)
	local ncols = data:size(4)
	print("Data size:"..nsamples..' '..nchannel..' '..nrows..' '..ncols)

	function dataset:size()
		return nsamples
	end

	local idx = 1
	local shuffle = torch.randperm(nsamples)

	function dataset:selectbatch()
		--all data have been used , restart
		if idx + batch > nsamples then
			shuffle = torch.randperm(nsamples)
			idx = 1
			print('data: shuffle the data')
		end
		local databatch = torch.Tensor()
		if batch > 1 then
			databatch:resize(batch, nchannel, nrows, ncols)
			for i=1,batch do
				local j = shuffle[idx]
				databatch[i] = data:select(1, j)
				idx = idx + 1
			end
		else
			local j = shuffle[idx]
			databatch = data:select(1, j)
			idx =idx + 1
		end
		return databatch
	end
	--[[
	setmetatable(dataset, {
		__index = function (self, index)
			return self:selectbatch()
		end
		})
	]]--
	return dataset
end
