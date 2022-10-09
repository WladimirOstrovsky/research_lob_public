import numpy as np


class Orderbook:
    """
    Holding basic functionalities of a limit order book
    """

    def __init__(self, periods, period_init, period_max):
        self.period_init = period_init
        self.period_max = period_max
        self.asks = np.array([])  # limit order book: [timestamp, lots, price, id]
        self.bids = np.array([])  # limit order book: [timestamp, lots, price, id]
        self.trades = np.array([])  # [timestamp, lots, price]
        # [timestamp (execution), lots, price, timestamp initiator, timestamp resting, id initiator, id resting]
        self.fills = np.array([])
        # [match_id, timestamp, side, lots, type, price, id]
        self.fills_orders = np.array([], dtype=object)
        self.price = np.zeros((periods, 2))  # combination of trades and average of bid-ask: [timestamp, price]
        self.ret = np.zeros((periods - 1, 2))  # [timestamp, price]
        self.tick_size = 0.01
        # commented out
        self.lob_store = np.array([])  # [timestamp stored, timestamp added, side, lots, price]
        # [timestamp, side, lots, type_, price, id]
        self.order_store = np.zeros((periods, 6), dtype=object)
        # [timestamp, side, lots, price, id] where timestamp is pointing
        self.order_cancel_store = np.array([], dtype=object)
        # [timestamp cleared, timestamp order, lots, price, id]
        self.order_cleared_store = np.array([], dtype=object)
        # to end of tick
        self.timestamp = 0  # starting timestamp
        # Integer identification
        # side
        #   buy: 1
        #   sell: -1
        # type_
        #   limit: 1
        #   market: -1

    @staticmethod
    def add_to_attribute(object_, attribute_name, array_):
        """
        Adds forecast to attribute store array.
        """
        attribute = getattr(object_, attribute_name)

        if attribute.size > 0:
            attribute = np.insert(attribute, len(attribute), array_, axis=0)
        else:
            attribute = array_

        setattr(object_, attribute_name, attribute)

    def add_price_history(self, price_history):
        """
        Sets price history as an attribute. Calculate return.
        Adjust timestamp so that every new order will start at new unique time
        :param price_history: np.array - [timestamp, price]
        """
        prices_array = np.column_stack((np.arange(1, len(price_history) + 1), price_history))
        returns = np.diff(np.log(prices_array[:, 1]))
        returns_array = np.column_stack((np.arange(2, len(price_history) + 1), returns))
        self.price[:len(prices_array)] = prices_array
        self.timestamp = len(prices_array)
        self.ret[:len(returns_array)] = returns_array

    def add_order(self, side, lots, type_, price, tif=None, id=None, remainder=False):
        """
        Adds order to orderbook and/or routes to match_order. Stores orders.
        [timestamp, direction, lots, price, id]
        :param side: string, buy (1) or sell (-1)
        :param lots: int
        :param type_: string, limit (1) or market (-1)  # TODO: change to non-negative ints: limit=1, market=2
        :param price: float
        :param tif: string
        :param tif: string (int as a string)
        :param remainder: boolean, True if function called by add_matched_order
        :return: np.array [timestamp, lots, price, timestamp initiator, timestamp resting]
        """
        # check whether add_order was called by match_order, i.e. happens when aggressor order is larger than
        # top-of-the book order
        if not remainder:
            self.timestamp = self.timestamp + 1
            print('')
            print('Tick #{}'.format(self.timestamp))

            # order = np.array([[self.timestamp, 1 if side == 'buy' else -1, lots, 1 if type_ == 'limit' else -1, price,
            #                    id]])
            order = np.array([[self.timestamp, side, lots, type_, price, id]], dtype=object)
            # self.order_store[self.timestamp - 1] = np.array([[self.timestamp, 1 if side == 'buy' else -1, lots,
            #                                                   1 if type_ == 'limit' else -1, price, id]])
            self.order_store[self.timestamp - 1] = np.array([[self.timestamp, side, lots, type_, price, id]],
                                                            dtype=object)
        else:
            # order = np.array([[self.timestamp, 1 if side == 'buy' else -1, lots, 1 if type_ == 'limit' else -1, price,
            #                    id]])
            order = np.array([[self.timestamp, side, lots, type_, price, id]], dtype=object)

        if type_ == 'limit':
            if side == 'buy':
                # check if asks exist
                if self.asks.size > 0:
                    # ask exist
                    # check if bid crosses ask
                    if order[0, 4] >= self.asks[0, 2]:
                        # execute order immediately
                        self.match_order(side, order, type_)
                    else:
                        # add bid
                        self.add_bid(order[:, [0, 2, 4, 5]])
                else:
                    # ask does not exist, add order to bids
                    self.add_bid(order[:, [0, 2, 4, 5]])

            elif side == 'sell':
                if self.bids.size > 0:
                    # bid exist
                    # check if ask crosses bid
                    if order[0, 4] <= self.bids[0, 2]:
                        # execute order immediately
                        self.match_order(side, order, type_)
                    else:
                        # add ask
                        self.add_ask(order[:, [0, 2, 4, 5]])
                else:
                    # bid does not exist, add order to asks
                    self.add_ask(order[:, [0, 2, 4, 5]])
            else:
                raise KeyError('Incorrect side.')

        elif type_ == 'market':
            self.match_order(side, order, type_)
        else:
            raise KeyError('Incorrect type.')

        fills_return = self.next_period(remainder)

        return fills_return

    def match_order(self, side, order, type_):
        """
        Routes to add_matched_order.
        :param side: string, buy or sell
        :param order: np.array
        :param type_: string, limit or market
        """
        if side == 'buy':
            if self.asks.size > 0:
                # create trade
                lots_filled = min(self.asks[0, 1], order[0, 2])
                order_lots_remaining = max(order[0, 2] - self.asks[0, 1], 0)
                lots_remaining = max(self.asks[0, 1] - order[0, 2], 0)
                price_filled = self.asks[0, 2]

                # get timestamps of both market participant's orders
                timestamp_initiator = order[0, 0]
                timestamp_resting = self.asks[0, 0]
                id_initiator = order[0, 5]
                id_resting = self.asks[0, 3]

                # add fill
                self.add_fill(lots_filled, price_filled, timestamp_initiator, timestamp_resting,
                              id_initiator, id_resting)

                # create match_id
                if self.fills_orders.size > 0:
                    # create next match_id
                    match_id = np.array([[self.fills_orders[:, 0].max() + 1]])
                else:
                    # start with zero match_id
                    match_id = np.array([[0]])

                # create arrays for orders that were filled
                fills_order_1 = np.concatenate((match_id, order), axis=1)
                # recover resting order
                fills_order_2 = np.concatenate((match_id, np.array([[timestamp_resting, 'sell', self.asks[0, 1],
                                                                     'limit', price_filled, id_resting]],
                                                                   dtype=object)), axis=1)
                fills_orders = np.vstack((fills_order_1, fills_order_2))

                # store filled orders
                if self.fills_orders.size > 0:
                    # append
                    self.fills_orders = np.insert(self.fills_orders, len(self.fills_orders), fills_orders, axis=0)
                else:
                    # insert
                    self.fills_orders = fills_orders

                # determine how many lots remaining and take relevant action
                if order[0, 2] == self.asks[0, 1]:
                    # remove ask
                    self.asks = np.delete(self.asks, 0, 0)
                elif order[0, 2] > self.asks[0, 1]:
                    # remove ask and check for more asks
                    self.asks = np.delete(self.asks, 0, 0)
                    order[0, 2] = order_lots_remaining
                    self.add_order(side, int(order[:, 2][0]), type_, float(order[:, 4][0]), tif=None, id=order[0, 5],
                                   remainder=True)
                elif order[0, 2] < self.asks[0, 1]:
                    # reduce ask size
                    self.asks[0, 1] = lots_remaining
                else:
                    pass
            else:
                self.add_bid(order)

        elif side == 'sell':
            if self.bids.size > 0:
                # create trade
                lots_filled = min(self.bids[0, 1], order[0, 2])
                order_lots_remaining = max(order[0, 2] - self.bids[0, 1], 0)
                lots_remaining = max(self.bids[0, 1] - order[0, 2], 0)
                price_filled = self.bids[0, 2]

                # get timestamps of both market participant's orders
                timestamp_initiator = order[0, 0]
                timestamp_resting = self.bids[0, 0]
                id_initiator = order[0, 5]
                id_resting = self.bids[0, 3]

                # add fill
                self.add_fill(lots_filled, price_filled, timestamp_initiator, timestamp_resting,
                              id_initiator, id_resting)

                # create match_id
                if self.fills_orders.size > 0:
                    # create next match_id
                    match_id = np.array([[self.fills_orders[:, 0].max() + 1]])
                else:
                    # start with zero match_id
                    match_id = np.array([[0]])

                # create arrays for orders that were filled
                fills_order_1 = np.concatenate((match_id, order), axis=1)
                # recover resting order
                fills_order_2 = np.concatenate((match_id, np.array([[timestamp_resting, 'buy', self.bids[0, 1],
                                                                     'limit', price_filled, id_resting]],
                                                                   dtype=object)), axis=1)
                fills_orders = np.vstack((fills_order_1, fills_order_2))

                # store filled orders
                if self.fills_orders.size > 0:
                    # append
                    self.fills_orders = np.insert(self.fills_orders, len(self.fills_orders), fills_orders, axis=0)
                else:
                    # insert
                    self.fills_orders = fills_orders

                # determine how many lots remaining and take relevant action
                if order[0, 2] == self.bids[0, 1]:
                    # remove ask
                    self.bids = np.delete(self.bids, 0, 0)
                elif order[0, 2] > self.bids[0, 1]:
                    # remove ask and check for more asks
                    self.bids = np.delete(self.bids, 0, 0)
                    order[0, 2] = order_lots_remaining
                    self.add_order(side, int(order[:, 2][0]), type_, float(order[:, 4][0]), tif=None, id=order[0, 5],
                                   remainder=True)
                elif order[0, 2] < self.bids[0, 1]:
                    # reduce bid size
                    self.bids[0, 1] = lots_remaining
                else:
                    pass
            else:
                self.add_ask(order)

        else:
            raise KeyError('Incorrect side.')

    def next_period(self, remainder=False):
        """
        Starts several functions after orders are added and/or matched.
        :param remainder: boolean
        """
        if not remainder:
            # remainder = False: hits only top-of-the-book

            # store bids and asks
            # self.add_to_lob_store()  # NOTE: this will slow down the simulation dramatically with exponential drag

            # sort order book
            self.sort_orderbook()

            # add trade
            self.add_trade()

            # save price
            self.add_to_price()
            # print('price: {}'.format(self.price[self.timestamp - 2:self.timestamp - 0]))

            # calculate returns
            self.add_to_return()

            # return fills from period
            if self.fills.size > 0:
                return {'fills': self.fills[self.fills[:, 0] == self.timestamp],
                        'fills_orders': self.fills_orders[np.isin(self.fills_orders[:, 0],
                                                                  self.fills_orders[
                                                                      self.fills_orders[:, 1] == self.timestamp][:,
                                                                  0])]}
            else:
                return {'fills': np.array([]), 'fills_orders': np.array([])}

        else:
            # remainder = True: hits multiple bids and asks
            self.sort_orderbook()

    def add_to_lob_store(self):
        """
        Adds bids and asks to limit order book store.
        """
        if self.asks.size > 0:
            order_stamped = np.insert(self.asks, 0, self.timestamp, axis=1)  # add timstamp
            order_stamped = np.insert(order_stamped, 2, -1, axis=1)  # add direction
            # check if lob_store is empty
            if self.lob_store.size > 0:
                # not empty
                self.lob_store = np.insert(self.lob_store, len(self.lob_store), order_stamped, axis=0)
            else:
                # empty
                self.lob_store = order_stamped
        else:
            pass

        if self.bids.size > 0:
            order_stamped = np.insert(self.bids, 0, self.timestamp, axis=1)  # add timestamp
            order_stamped = np.insert(order_stamped, 2, 1, axis=1)  # add direction
            # check if lob_store is empty
            if self.lob_store.size > 0:
                # not empty
                self.lob_store = np.insert(self.lob_store, len(self.lob_store), order_stamped, axis=0)
            else:
                # empty
                self.lob_store = order_stamped
        else:
            pass

    def sort_orderbook(self):
        """
        Sorts order book by price/time preference.
        """
        if self.asks.size > 0:
            sort_first = self.asks[np.argsort(self.asks[:, 0])]  # sort by timestamp
            self.asks = sort_first[np.argsort(sort_first[:, 2])]  # sort by price ascending
        else:
            pass

        if self.bids.size > 0:
            sort_first = self.bids[np.argsort(self.bids[:, 0])]  # sort by timestamp
            self.bids = sort_first[np.argsort(sort_first[:, 2])][::-1]  # sort by price descending
        else:
            pass

    def add_average_price(self):
        """
        Calculates average price and route to add_price for price to be stored.
        """
        if self.asks.size > 0 and self.bids.size > 0:
            # average of bid and ask
            average_price = (self.asks[0][2] + self.bids[0][2]) / 2
            self.price[self.timestamp - 1] = np.array([[self.timestamp, average_price]])
        elif self.asks.size > 0 and not self.bids.size > 0:
            # only ask
            self.price[self.timestamp - 1] = np.array([[self.timestamp, self.asks[0][2]]])
        elif not self.asks.size > 0 and self.bids.size > 0:
            # only bid
            self.price[self.timestamp - 1] = np.array([[self.timestamp, self.bids[0][2]]])
        else:
            # no ask and no bid
            # if self.trades.size > 0:
                # past trade exist
            # use last price as current price
            self.price[self.timestamp - 1] = self.price[self.timestamp - 2]
            # else:
            #     no trades exist, insert -1 as indication
                # self.price[self.timestamp - 1] = np.array([[self.timestamp, -1]])

    def add_to_price(self):
        """
        Routes to add_average_price or adds recent trade to prices.
        """
        if self.trades.size > 0:
            # use trade as last price
            if self.trades[-1][0] == self.timestamp:
                # mask = self.trades[:, 0] == self.timestamp
                # if len(self.trades[mask]) > 1:
                #     print('HERE')
                #     len(self.trades)
                #     self.price[self.trades[:, 0] + 1, :]
                #     self.trades[:, 2] - np.take(self.price[:, 1], (self.trades[:, 0] - 1).astype(int))
                # last trade timestamp coincides with current timestamp
                self.price[self.timestamp - 1] = self.trades[-1, [0, 2]]
                # print(self.trades[-1, [0, 2]])
            else:
                # no recent trade, add average price
                self.add_average_price()
        else:
            self.add_average_price()

        # print statements
        # print('price:          {:.1f}'.format(self.price[self.timestamp - 1][1]))
        # print('bids:    {}     {:.1f}'.format(int(self.bids[0][0]), self.bids[0][2]))
        # print('asks:    {}     {:.1f}'.format(int(self.asks[0][0]), self.asks[0][2]))

    def add_to_return(self):
        """
        Calculates return and stores it.
        """
        # timestamp - 2 must at least be zero - enough prices to calculate return
        if self.timestamp - 2 >= 0:
            price_t = self.price[self.timestamp - 2]
            price_t_1 = self.price[self.timestamp - 1]
            ret = np.diff(np.log(self.price[self.timestamp - 2:self.timestamp, 1]))[0]
            # if abs(ret) > 5/100:
            #     print('HERE')
            self.ret[self.timestamp - 2] = np.array([[self.timestamp, ret]])
        else:
            pass

    def add_ask(self, order):
        """
        Adds ask to orderbook (asks)
        [timestamp, lots, order]
        :param order: np.array
        """
        if self.asks.size > 0:
            self.asks = np.insert(self.asks, len(self.asks), order, axis=0)
        else:
            self.asks = order

    def add_bid(self, order):
        """
        Adds bid to orderbook (bids)
        [timestamp, lots, order]
        :param order: np.array
        """
        if self.bids.size > 0:
            self.bids = np.insert(self.bids, len(self.bids), order, axis=0)
        else:
            self.bids = order

    def add_fill(self, lots_filled, price_filled, timestamp_initiator, timestamp_resting, id_initiator, id_resting):
        """
        Adds fills to fills store.
        [timestamp, lots, price, timestamp initiator, timestamp resting, id initiator, id resting]
        :param lots_filled: int
        :param price_filled: float
        :param timestamp_initiator: int
        :param timestamp_resting: int
        :param id_initiator: int
        :param id_resting: int
        """
        if self.fills.size > 0:
            self.fills = np.insert(self.fills, len(self.fills), np.array([[self.timestamp, lots_filled, price_filled,
                                                                           timestamp_initiator, timestamp_resting,
                                                                           id_initiator, id_resting]], dtype=object),
                                   axis=0)
        else:
            self.fills = np.array([[self.timestamp, lots_filled, price_filled, timestamp_initiator, timestamp_resting,
                                    id_initiator, id_resting]], dtype=object)

    def add_trade(self):
        """
        Calculates volume-weighted average price for trades and stores it.
        [timestamp, lots, vwap]
        """
        if self.fills.size > 0:
            # add only trade if happened in at current timestamp
            if self.fills[-1][0] == self.timestamp:
                # calculate vwap
                last_fills_timestamp = self.fills[-1, 0]
                last_fills = self.fills[np.where(self.fills[:, 0] == last_fills_timestamp)]
                last_fills_volume = np.sum(last_fills[:, 1])
                vwap = np.sum(last_fills[:, 1] * last_fills[:, 2]) / np.sum(last_fills[:, 1])

                # add trade to store
                if self.trades.size > 0:
                    self.trades = np.insert(self.trades, len(self.trades),
                                            np.array([[self.timestamp, last_fills_volume, vwap]]), axis=0)
                else:
                    self.trades = np.array([[self.timestamp, last_fills_volume, vwap]])
            else:
                pass
        else:
            pass

    def clean_orderbook(self):
        """
        Removes all bids and asks. Store them.
        """
        # store asks
        if self.asks.size > 0:
            direction_asks = np.insert(self.asks, 1, -1, axis=1)  # add direction
            if self.order_cancel_store.size > 0:
                self.order_cancel_store = np.insert(self.order_cancel_store, len(self.order_cancel_store),
                                                    direction_asks, axis=0)
            else:
                self.order_cancel_store = direction_asks
        else:
            pass

        # store bids
        if self.bids.size > 0:
            direction_bids = np.insert(self.bids, 1, 1, axis=1)  # add direction
            if self.order_cancel_store.size > 0:
                self.order_cancel_store = np.insert(self.order_cancel_store, len(self.order_cancel_store),
                                                    direction_bids, axis=0)
            else:
                self.order_cancel_store = direction_bids
        else:
            pass

        # clear the books
        self.asks = np.array([])
        self.bids = np.array([])

    def cancel_order(self, timestamp, modify, remainder=False):
        """
        Cancels order from orderbook. Timestamp is used as an ID. Store it.
        :param timestamp: int
        :param modify: boolean, triggered by modify_order()=True
        :param remainder: boolean, default is false which should be the case for pure cancellations as the order is
        removed from the LOB. modify_order will also use cancel_order, however, order is replaced meaning that remainder
        should be True to avoid running next_period() completely, i.e. adding to prices etc.
        """
        if modify:
            pass
        else:
            self.timestamp = self.timestamp + 1
            print('')
            print('Tick #{}'.format(self.timestamp))

        # find matching bid
        order_found_bid_id = np.where(self.bids[:, 0] == timestamp)
        if order_found_bid_id[0].size > 0:
            # store cancelled order
            direction_bid = np.insert(self.bids[order_found_bid_id[0][0]], 1, 1, axis=0)
            self.add_to_attribute(self, 'order_cancel_store', np.array([direction_bid]))

            # drop order
            self.bids = np.delete(self.bids, order_found_bid_id[0][0], axis=0)
        else:
            pass

        # find matching ask
        order_found_ask_id = np.where(self.asks[:, 0] == timestamp)
        if order_found_ask_id[0].size > 0:
            # store cancelled order
            direction_ask = np.insert(self.asks[order_found_ask_id[0][0]], 1, -1, axis=0)
            self.add_to_attribute(self, 'order_cancel_store', np.array([direction_ask]))

            # drop order
            self.asks = np.delete(self.asks, order_found_ask_id[0][0], axis=0)
        else:
            pass

        # fills_return = self.next_period(remainder=False)
        fills_return = self.next_period(remainder=remainder)

        return fills_return

    def modify_order(self, timestamp, side, lots, type_, price, tif=None, id=None):
        """
        Modifies order by cancelling first and adding new order.
        :param timestamp: int
        :param side: string, buy or sell
        :param lots: int
        :param type_: string, limit or market
        :param price: float
        :param tif: string
        """
        self.cancel_order(timestamp, modify=True, remainder=True)
        fills_return = self.add_order(side, lots, type_, price, tif, id)

        return fills_return

    def add_random_orders(self, tau, lot_size=None, mult=1, interval=(800, 1200)):
        """
        Add random orders and store them with appropriate timestamps
        :param tau: int
        :param lot_size: tuple, lower and upper bound for lot size, per default None
        :param mult: int, per default 1
        :param interval: tuple
        :return:
        """
        # create random prices, number of random prices is tau * 2 - 10, 10 will be best bids and asks
        # order_prices_temp = np.round(np.random.uniform(interval[0], interval[1], tau * 2 - 10), 1)
        order_prices_temp = np.round(np.random.normal(loc=1000, scale=50, size=tau * mult - 10), 1)
        best_asks = np.ones((5,)) * 1000.10
        best_bids = np.ones((5,)) * 999.90
        order_prices = np.concatenate((best_bids, best_asks, order_prices_temp))
        if lot_size:
            orders = np.column_stack((np.arange(self.timestamp - tau * mult + 1, self.timestamp + 1),
                                      np.random.randint(lot_size[0], lot_size[1], size=(tau * mult, 1)),
                                      order_prices, [None] * tau * mult))
        else:
            orders = np.column_stack((np.arange(self.timestamp - tau * mult + 1, self.timestamp + 1),
                                      np.ones((tau * mult, 1)),
                                      order_prices, [None] * tau * mult))

        # sort orders and split into bids and asks
        orders = orders[np.argsort(orders[:, 2])]
        asks = orders[orders[:, 2] > 1000.]
        bids = orders[orders[:, 2] <= 1000.]

        # store in attributes
        self.asks = asks
        self.bids = bids
        self.sort_orderbook()

    def clean_last_orders(self, tau):
        """
        Clears last tau orders from bids and asks.
        :param tau: int
        """
        # TODO: need to take care of case where all orders are removed, will not work removing all, need to keep at least 1 as in clean_old_orders
        # TODO: something wrong
        # pre-define np.array
        cleared_orders = np.array([])
        mask_timestamps = np.sort(np.concatenate((self.bids[:, 0], self.asks[:, 0])))[0:tau]

        if mask_timestamps.shape[0] > 0:

            # mask orders to be cleared
            mask = np.isin(self.bids[:, 0], mask_timestamps)
            if mask.all() & (len(self.bids) > 0):
                mask[-1] = False

            # insert cleared orders
            cleared_orders = np.column_stack((
                np.ones((self.bids[mask].shape[0])) * self.timestamp,
                self.bids[mask]))

            # drop bids
            self.bids = self.bids[~mask]

        if mask_timestamps.shape[0] > 0:

            # mask orders to be cleared
            mask = np.isin(self.asks[:, 0], mask_timestamps)
            if mask.all() & (len(self.asks) > 0):
                mask[-1] = False

            # insert cleared orders
            if cleared_orders.size > 0:
                cleared_orders = np.insert(cleared_orders, len(cleared_orders),
                                           np.column_stack((
                                               np.ones((self.asks[mask].shape[
                                                   0])) * self.timestamp,
                                               self.asks[mask])),
                                           axis=0)
            else:
                cleared_orders = np.column_stack((
                    np.ones((self.asks[mask].shape[0])) * self.timestamp,
                    self.asks[mask]))

            # drop asks
            self.asks = self.asks[~mask]

        # add to attribute
        if cleared_orders.size > 0:
            self.add_to_attribute(self, 'order_cleared_store', cleared_orders)
        else:
            pass

        return cleared_orders

    def clean_oldest_orders(self, tau):
        """
        Clears oldest orders, older than tau periods.
        :param tau: int
        :return: np.array [timestamp, timestamp cleared, , lots, price]
        """
        # pre-define np.array
        cleared_orders = np.array([])

        # bids
        mask = self.bids[:, 0] < self.timestamp - tau
        if (mask.sum() == len(self.bids)) & (len(self.bids) > 0):
            mask[-1] = False

        if mask.sum() > 0:
            # insert cleared orders
            cleared_orders = np.column_stack((np.ones((len(self.bids[mask]))) * self.timestamp, self.bids[mask]))

        # drop bids
        self.bids = self.bids[~mask]

        # asks
        mask = self.asks[:, 0] < self.timestamp - tau
        if (mask.sum() == len(self.asks)) & (len(self.asks) > 0):
            mask[-1] = False

        if mask.sum() > 0:
            # insert cleared orders
            if cleared_orders.size > 0:
                cleared_orders = np.insert(cleared_orders, len(cleared_orders),
                                           np.column_stack((np.ones((len(self.asks[mask]))) * self.timestamp,
                                                            self.asks[mask])),
                                           axis=0)
            else:
                cleared_orders = np.column_stack((np.ones((len(self.asks[mask]))) * self.timestamp, self.asks[mask]))

        # drop asks
        self.asks = self.asks[~mask]

        # add to attribute
        if cleared_orders.size > 0:
            self.add_to_attribute(self, 'order_cleared_store', cleared_orders)
        else:
            pass

        return cleared_orders

    def __getinitargs__(self):
        # we need this function to load the saved pickle file
        return self.args

    def __getnewargs__(self):
        return self.period_init, self.period_max

    def __getstate__(self):
        # this function saves the state of the initialized
        state = self.__dict__.copy()
        return state


if __name__ == '__main__':
    orderbook = Orderbook(periods=6)
    orderbook.add_order('buy', 5, 'limit', 10.)
    orderbook.add_order('sell', 5, 'limit', 15.)
    orderbook.add_order('buy', 5, 'limit', 12.)
    orderbook.add_order('sell', 5, 'limit', 16.)
    orderbook.add_order('buy', 11, 'limit', 16.)
    orderbook.add_order('sell', 5, 'limit', 14.)
