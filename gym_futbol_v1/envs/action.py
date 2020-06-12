import enum
from .helper import Side, get_vec, ball_move_with_player


class ArrowKeys(enum.Enum):
    NOOP = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class ActionKeys(enum.Enum):
    NOOP = 0
    DASH = 1
    SHOOT = 2
    PRESS = 3
    PASS = 4


def action_key_string(action_key):
    if action_key == 0:
        return "noop "
    elif action_key == 1:
        return "dash "
    elif action_key == 2:
        return "shoot"
    elif action_key == 3:
        return "press"
    elif action_key == 4:
        return "pass "


def arrow_key_string(arrow_key):
    if arrow_key == 0:
        return "noop "
    elif arrow_key == 1:
        return "up   "
    elif arrow_key == 2:
        return "right"
    elif arrow_key == 3:
        return "down "
    elif arrow_key == 4:
        return "left "


def process_action(self, player, action):
    """
    Process the action for each player
    """
    # Arrow Keys: NOOP
    if action[0] == 0:
        force_x, force_y = 0, 0
    # Arrow Keys: UP
    elif action[0] == 1:
        force_x, force_y = 0, 1
    # Arrow Keys: RIGHT
    elif action[0] == 2:
        force_x, force_y = 1, 0
    # Arrow Keys: DOWN
    elif action[0] == 3:
        force_x, force_y = 0, -1
    # Arrow Keys: LEFT
    elif action[0] == 4:
        force_x, force_y = -1, 0
    else:
        print("invalid arrow keys")

    # Action keys
    # noop [0]
    if action[1] == 0:
        player.apply_force_to_player(self.PLAYER_WEIGHT * force_x,
                                     self.PLAYER_WEIGHT * force_y)

        ball_move_with_player(self.ball, player)

    # dash [1]
    elif action[1] == 1:
        player.apply_force_to_player(self.PLAYER_FORCE_LIMIT * force_x,
                                     self.PLAYER_FORCE_LIMIT * force_y)
        ball_move_with_player(self.ball, player)

    # shoot [2]
    elif action[1] == 2:
        if self.ball.has_contact_with(player):
            if player.side == Side("left"):
                goal = [self.WIDTH, self.HEIGHT/2]
            elif player.side == Side("right"):
                goal = [0, self.HEIGHT/2]
            else:
                print("invalid side")

            ball_pos = self.ball.get_position()
            ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                goal, ball_pos)

            ball_force_x = self.BALL_FORCE_LIMIT * \
                ball_to_goal_vec[0] / ball_to_goal_vec_mag
            ball_force_y = self.BALL_FORCE_LIMIT * \
                ball_to_goal_vec[1] / ball_to_goal_vec_mag

            # decrease the velocity influence on shoot
            self.ball.body.velocity /= 2

            self.ball_owner_side = player.side
            self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
        else:
            pass

    # press [3]
    elif action[1] == 3:
        # cannot press with ball
        if self.ball.has_contact_with(player):
            player.apply_force_to_player(0, 0)
        # no ball, no arrow keys, run to ball (press)
        elif action[0] == 0:
            ball_pos = self.ball.get_position()
            player_pos = player.get_position()

            player_to_ball_vec, player_to_ball_vec_mag = get_vec(
                ball_pos, player_pos)

            player_force_x = self.PLAYER_FORCE_LIMIT * \
                player_to_ball_vec[0] / player_to_ball_vec_mag
            player_force_y = self.PLAYER_FORCE_LIMIT * \
                player_to_ball_vec[1] / player_to_ball_vec_mag

            player.apply_force_to_player(player_force_x, player_force_y)
        # no ball, arrow keys pressed, run as the arrow key, similar to dash
        else:
            player.apply_force_to_player(self.PLAYER_FORCE_LIMIT * force_x,
                                         self.PLAYER_FORCE_LIMIT * force_y)
            ball_move_with_player(self.ball, player)

    # pass [4]
    elif action[1] == 4:
        if self.ball.has_contact_with(player):
            team = self.team_A if player.side == Side("left") else self.team_B

            target_player = team.get_pass_target_teammate(
                player, arrow_keys=action[0])

            goal = target_player.get_position()

            ball_pos = self.ball.get_position()
            ball_to_goal_vec, ball_to_goal_vec_mag = get_vec(
                goal, ball_pos)

            ball_force_x = (self.BALL_FORCE_LIMIT - 20) * \
                ball_to_goal_vec[0] / ball_to_goal_vec_mag
            ball_force_y = (self.BALL_FORCE_LIMIT - 20) * \
                ball_to_goal_vec[1] / ball_to_goal_vec_mag

            # decrease the velocity influence on pass
            self.ball.body.velocity /= 10

            self.ball_owner_side = player.side
            self.ball.apply_force_to_ball(ball_force_x, ball_force_y)
        # cannot pass ball without ball
        else:
            pass

    else:
        print("invalid action key")
