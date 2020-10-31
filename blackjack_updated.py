#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:18:17 2020

@author: joshuageiser
"""

import os
import numpy as np
import pandas as pd
import random

class Params():
    def __init__(self):
        # 'input', 'random_policy', 'fixed_policy'
        self.action_type = 'random_policy'
        
        # Only used for 'random_policy' or 'fixed_policy' input
        self.num_games = 10000
        
        # Filepath to fixed policy file (only used for 'fixed_policy' input)
        self.fixed_policy_filepath = os.path.join(os.getcwd(), 'test_policy.policy')
        
        return

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card():
    return random.sample(deck, 1)[0]


def draw_hand():
    return [draw_card(), draw_card()]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def player_won(player, dealer):
    if is_bust(player):
        return False
    elif is_bust(dealer):
        return True
    elif sum_hand(player) > sum_hand(dealer):
        return True
    else:
        return False

def hand_to_state(player):
    return sum_hand(player) - 1

def get_reward(state, action):
    return 0

class BlackJack_game():
    def __init__(self, params):
        self.player = draw_hand()
        self.dealer = [draw_card()]
        self.sarsp = []
        self.sarsp_arr = np.array([], dtype='int').reshape(0,4)
        
        self.action_type = params.action_type # 'input', 'random_policy', 'fixed_policy'
        self.verbose = (params.action_type == 'input')
        self.num_games = params.num_games
        self.fixed_policy_filepath = params.fixed_policy_filepath
        self.policy = self.load_policy()
        
        self.lose_state = 0
        self.win_state = 1
        self.terminal_state = 2
        
        self.lose_reward = -10
        self.win_reward = 10
        return
    
    def reset(self): 
        self.player = draw_hand()
        self.dealer = [draw_card()]
        self.sarsp = []
        return
    
    def load_policy(self):
        if self.action_type in ['random_policy', 'input']:
            return None
        
        f = open(self.fixed_policy_filepath, 'r')
        
        data = f.read()
        data = data.split()
        
        policy = [int(x) for x in data]
        
        return policy
        
    def print_iter(self):
        if not self.verbose:
            return
        
        print(f'Player hand: {self.player}\t\t sum: {sum_hand(self.player)}')
        print(f'Dealer hand: {self.dealer}\t\t sum: {sum_hand(self.dealer)}')
        return
    
    def get_action(self, state):
        if self.action_type == 'input':
            action = int(input('Hit (1) or Pass (0): '))
        elif self.action_type == 'random_policy':
            action = np.random.randint(2)
        elif self.action_type == 'fixed_policy':
            action = self.policy[state]
        return action
        
    
    def play_game(self):
        
        if self.verbose:
            print('New Game!\n')
        
        done = False
        while(not done):
            
            self.print_iter()
            state = hand_to_state(self.player)
            action = self.get_action(state)
            reward = get_reward(state, action)
            
            if action:  # hit: add a card to players hand and return
                self.player.append(draw_card())
                if is_bust(self.player):
                    done = True
                else:
                    done = False
            else:  # stick: play out the dealers hand, and score
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card())
                done = True
            
            if(not done):
                sp = hand_to_state(self.player)
                self.sarsp.append([state, action, reward, sp])
                
        self.print_iter()
        player_won_bool = player_won(self.player, self.dealer)
        if player_won_bool:
            sp = self.win_state
        else:
            sp = self.lose_state
        self.sarsp.append([state, action, reward, sp])
        
        # Add a row with 0 action, 0 reward and terminal state for next state
        state = sp
        if player_won_bool:
            reward = self.win_reward
        else:
            reward = self.lose_reward
        self.sarsp.append([state, np.random.randint(2), reward, self.terminal_state])
            
        if self.verbose:
            print(f'Player won?: {player_won_bool}')
        
        # Append current run data to full sarsp_arr 
        self.sarsp_arr = np.vstack((self.sarsp_arr, np.array(self.sarsp)))
    
        return
    
    def output_sarsp_file(self):
        output_filepath = os.path.join(os.getcwd(), 'random_policy_runs.txt')
        header = ['s', 'a', 'r', 'sp']
        pd.DataFrame(self.sarsp_arr).to_csv(output_filepath, header=header, index=None)
        return
    
    def print_stats(self):
        num_wins = np.count_nonzero(self.sarsp_arr[:,0] == self.win_state)
        num_lose = np.count_nonzero(self.sarsp_arr[:,0] == self.lose_state)
        
        print(f'Number of games: {self.num_games}')
        print(f'Number of wins: {num_wins}')
        print(f'Number of losses: {num_lose}')
        print(f'Win Percentage: {num_wins / self.num_games : .2f}')        
        
        return
    
    def play_games(self):
        
        for i in range(self.num_games):
            self.play_game()
            self.reset()
        
        print(self.sarsp_arr)
        self.print_stats()
        
        if self.action_type == 'random_policy':
            self.output_sarsp_file()
            
        return
    

    

def main(): 
    
    params = Params()
    
    game = BlackJack_game(params)
    
    if params.action_type == 'input':
        game.play_game()
    else:
        game.play_games()
    
    return

if __name__ == "__main__":
   main()