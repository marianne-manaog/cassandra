/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.cassandra.io.tries;

import java.io.IOException;

public interface TrieSerializer<Value, Dest>
{
    int sizeofNode(SerializationNode<Value> node, long nodePosition);
    default int inpageSizeofNode(SerializationNode<Value> node, long nodePosition)
    {
        return sizeofNode(node, nodePosition);
    }

    // Only called after all children's serializedPositions have been set.
    void write(Dest dest, SerializationNode<Value> node, long nodePosition) throws IOException;
    default void inpageWrite(Dest dest, SerializationNode<Value> node, long nodePosition) throws IOException
    {
        write(dest, node, nodePosition);
    }
}