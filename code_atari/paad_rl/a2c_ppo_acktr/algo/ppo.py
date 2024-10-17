import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 beta=False,
                 imitate=False):

        self.actor_critic = actor_critic
        self.beta = beta

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        
    def unset_imitate(self, lr=None, eps=None, alpha=None):
        return

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        penalty_advantages=rollouts.penalty_returns[:-1] - rollouts.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        penalty_advantages = (penalty_advantages - penalty_advantages.mean()) / (
            penalty_advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        value_loss_penalty_epoch = 0
        action_loss_penalty_epoch = 0
        dist_entropy_penalty_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, penalty_advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, penalty_advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, penalty_return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, penalty_adv_targ, probs_batch, old_prob_log_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, beta=self.beta)
                
                _,prob_log, prob_entropy,_=self.actor_critic.evaluate_probs(obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()


                ratio_penalty=torch.exp(prob_log -
                                  old_prob_log_batch)
                surr1_penalty = ratio_penalty * penalty_adv_targ
                surr2_penalty = torch.clamp(ratio_penalty, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * penalty_adv_targ
                action_loss_penalty = -torch.min(surr1_penalty, surr2_penalty).mean()


                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                    

                    value_losses_penalty = (values - penalty_return_batch).pow(2)
                    value_losses_clipped_penalty = (
                        value_pred_clipped - penalty_return_batch).pow(2)
                    value_loss_penalty = 0.5 * torch.max(value_losses_penalty,
                                                 value_losses_clipped_penalty).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    value_loss_penalty=0.5 * (penalty_return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()


                loss_adv=value_loss * self.value_loss_coef + action_loss -dist_entropy * self.entropy_coef

                (loss_adv).backward(retain_graph=True)

                for param in self.actor_critic.base.main.parameters():
                    param.requires_grad = False
                w_1=0.8
                w_2=0.2
                loss_penalty=value_loss_penalty * self.value_loss_coef + action_loss_penalty -prob_entropy * self.entropy_coef
  
                (w_1*loss_adv+w_2*loss_penalty).backward()

                 # Unfreeze the backbone parameters for future updates
                for param in self.actor_critic.base.main.parameters():
                    param.requires_grad = True                       
                                                                                    
                
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                
                #for name, param in self.actor_critic.named_parameters():
                #    if param.requires_grad:  # Only print trainable parameters
                #        print(f"{name}: {param.data.shape}")

                
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                value_loss_penalty_epoch += value_loss_penalty.item()
                action_loss_penalty_epoch += action_loss_penalty.item()
                dist_entropy_penalty_epoch += prob_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        value_loss_penalty_epoch /= num_updates
        action_loss_penalty_epoch /= num_updates
        dist_entropy_penalty_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, value_loss_penalty_epoch, action_loss_penalty_epoch, dist_entropy_penalty_epoch
