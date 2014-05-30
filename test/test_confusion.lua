require 'torch'
require 'optim'

n_feature = 3
classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

print'ConfusionMatrix:__init() test'
cm = optim.ConfusionMatrix(#classes, classes)

target = 3
prediction = torch.randn(#classes)

print'ConfusionMatrix:add() test'
cm:add(prediction, target)

batch_size = 8

-- add batch
targets = torch.randperm(batch_size)
predictions = torch.randn(batch_size, #classes)

print'ConfusionMatrix:batchAdd() test'
cm:batchAdd(predictions, targets)
assert(cm.mat:sum() == batch_size + 1, 'missing examples')

-- add batch size 1
targets = torch.FloatTensor({5})
predictions = torch.randn(1, #classes)

print'ConfusionMatrix:batchAdd() test 2, batch size 1'
cm.prediction_batch = nil -- because we are using different batchsize
cm:batchAdd(predictions, targets)
assert(cm.mat:sum() == batch_size + 2, 'missing examples')

print'ConfusionMatrix:updateValids() test'
cm:updateValids()

print'ConfusionMatrix:__tostring__() test'
print(cm)
