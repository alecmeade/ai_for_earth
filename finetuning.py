# Fine tune with back prop
# TODO move to finetuning directory
for epoch in range(params['max_epochs']):
    finetuning_loss, finetuning_reports = train(model, optimizer, loss_fn,
                                                finetuning_loader, metrics, params['log_steps'], device)
    
    test_loss, test_reports = evaluate(model, loss_fn, test_loader, metrics, device)
    
# Finetine with Dropout
dropout_masks = {
    'start': [256, 256],
#     'down_0': None,
#     'down_1': None,
#     'down_2': None,
#     'down_3': None,
#     'down_4': None,
#     'up_0': None,
#     'up_1': None,
#     'up_2': None,
#     'up_3': None,
#     'end': None,   
}

finetuning_params = {
    "n_generations": 100
    "n_children": 10
}



evolver = MatrixEvolver([m for k, m in dropout_masks.items() if m is not None],
                        CrossoverType.UNIFORM, MutationType.FLIP_BIT)

def dropout_finetune(engine, batch):
    model.eval()
    with torch.no_grad():
        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
        for child in range(params['n_children']):
            child_masks = evolver.spawn_child()
            model.set_dropout_masks({k: torch.tensor(child_masks[i], 
                                                     device=device,
                                                     dtype=torch.float) for i, k in enumerate(dropout_masks.keys())})

            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()
            evolver.add_child(child_masks, total_loss)

        evolver.update_parents()

    return pass


for epoch in range(finetuning_params['max_epochs']):
    dropout_finetune(model, loss_fn, data_loader, metrics, dropout_masks, params, device)
    test_loss, test_reports = evaluate(model, loss_fn, test_loader, metrics, device)
